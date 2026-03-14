"""Build control presets via clustering + classification/regression.

Methods:
    classify (default): HDBSCAN clustering → RandomForest classifier
    regress: RandomForest regressor for direct control prediction
    sweep: KMeans K sweep (K=3,5,7,10) with automatic best-K selection

Pipeline (classify):
    1. Load train pieces → build windows with oracle controls
    2. HDBSCAN on [N, 2] normalized oracle controls → K clusters
    3. Noise (label=-1) reassigned to nearest centroid
    4. K preset centroids = cluster-wise mean of normalized controls
    5. Extract 21-dim window features → train RandomForest classifier
    6. Optionally evaluate on val set
    7. Save presets.json, classifier.joblib, preset_report.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from mvi_v3.config import BaselineConfig
from mvi_v3.data.events import DatasetStats
from mvi_v3.data.features import add_derived_features, sort_and_reindex
from mvi_v3.data.ingest import load_piece_directory
from mvi_v3.data.preset_features import extract_window_features
from mvi_v3.data.windowing import build_windows, normalize_oracle_controls
from mvi_v3.io.artifacts import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build control presets from training data.")
    parser.add_argument("--train-dir", required=True, help="Path to training MIDI pieces")
    parser.add_argument("--stats", required=True, help="Path to stats.json from training")
    parser.add_argument("--output-dir", required=True, help="Directory to save presets + classifier")
    parser.add_argument("--val-dir", default=None, help="Optional validation set for classifier evaluation")
    parser.add_argument("--method", default="classify", choices=["classify", "regress", "sweep"],
                        help="classify: HDBSCAN+RF classifier, regress: RF regressor, sweep: KMeans K sweep")
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--time-scale", type=float, default=1.0)
    return parser.parse_args()


def _prepare_pieces(path: str, config: BaselineConfig) -> list[list]:
    pieces = load_piece_directory(path, time_scale=config.time_scale)
    return [add_derived_features(sort_and_reindex(piece), config) for piece in pieces]


def _extract_features_matrix(windows: list) -> np.ndarray:
    return np.stack([extract_window_features(w) for w in windows])


def _load_stats(stats_path: str) -> DatasetStats:
    stats_payload = load_json(stats_path)
    return DatasetStats(
        feature_means={k: float(v) for k, v in stats_payload["feature_means"].items()},
        feature_stds={k: float(v) for k, v in stats_payload["feature_stds"].items()},
        velocity_min=float(stats_payload["velocity_min"]),
        velocity_max=float(stats_payload["velocity_max"]),
        oracle_mins=stats_payload.get("oracle_mins", []),
        oracle_maxs=stats_payload.get("oracle_maxs", []),
    )


def _load_train_data(args, stats: DatasetStats):
    """Load training windows and return (windows, oracle_controls, X_features)."""
    config = BaselineConfig(time_scale=args.time_scale, enable_controls=True)
    print("Loading training pieces...")
    train_pieces = _prepare_pieces(args.train_dir, config)
    train_windows = build_windows(train_pieces, stats, config)
    print(f"  {len(train_windows)} windows from {len(train_pieces)} pieces")

    normalize_oracle_controls(train_windows, stats.oracle_mins, stats.oracle_maxs)
    oracle_controls = np.stack([w.oracle_controls for w in train_windows])
    print(f"  Oracle controls shape: {oracle_controls.shape}")

    print("Extracting window features...")
    X_train = _extract_features_matrix(train_windows)
    print(f"  Feature matrix: {X_train.shape}")
    return train_windows, oracle_controls, X_train, config


def _load_val_data(args, stats: DatasetStats, config: BaselineConfig):
    """Load val windows and return (windows, oracle_controls, X_features) or None."""
    if not args.val_dir:
        return None
    print("\nLoading validation pieces...")
    val_pieces = _prepare_pieces(args.val_dir, config)
    val_windows = build_windows(val_pieces, stats, config)
    normalize_oracle_controls(val_windows, stats.oracle_mins, stats.oracle_maxs)
    X_val = _extract_features_matrix(val_windows)
    val_oracle = np.stack([w.oracle_controls for w in val_windows])
    return val_windows, val_oracle, X_val


def _run_classify(args, output_dir: Path, stats: DatasetStats) -> None:
    """Original HDBSCAN clustering + RF classifier pipeline."""
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib

    train_windows, oracle_controls, X_train, config = _load_train_data(args, stats)

    # --- HDBSCAN clustering ---
    print(f"Running HDBSCAN (min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples})...")
    clusterer = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    labels = clusterer.fit_predict(oracle_controls)

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    noise_ratio = n_noise / len(labels)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points ({noise_ratio:.1%})")

    # Fallback: if HDBSCAN gives K<=1, use KMeans
    used_fallback = False
    if n_clusters <= 1:
        warnings.warn(
            f"HDBSCAN found {n_clusters} cluster(s). Falling back to KMeans(K=5).",
            stacklevel=1,
        )
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = kmeans.fit_predict(oracle_controls)
        n_clusters = 5
        n_noise = 0
        noise_ratio = 0.0
        used_fallback = True
        print(f"  KMeans fallback: {n_clusters} clusters")

    # Reassign noise to nearest centroid
    unique_labels = sorted(set(labels) - {-1})
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[label] = oracle_controls[mask].mean(axis=0)

    if n_noise > 0 and not used_fallback:
        centroid_array = np.stack([centroids[l] for l in unique_labels])
        noise_mask = labels == -1
        noise_points = oracle_controls[noise_mask]
        dists = np.linalg.norm(noise_points[:, None, :] - centroid_array[None, :, :], axis=2)
        nearest = np.array(unique_labels)[dists.argmin(axis=1)]
        labels[noise_mask] = nearest
        print(f"  Reassigned {n_noise} noise points to nearest centroids")
        for label in unique_labels:
            mask = labels == label
            centroids[label] = oracle_controls[mask].mean(axis=0)

    # Build preset list (remap labels to 0..K-1)
    label_remap = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_remap[l] for l in labels])
    presets = []
    cluster_sizes = []
    for i in range(len(unique_labels)):
        old_label = unique_labels[i]
        presets.append(centroids[old_label].tolist())
        cluster_sizes.append(int((labels == i).sum()))

    print(f"\n  Preset centroids:")
    for i, (c, s) in enumerate(zip(presets, cluster_sizes)):
        print(f"    [{i}] expressiveness={c[0]:.3f}, dynamics_center={c[1]:.3f}  (n={s})")

    # Train classifier
    y_train = labels
    print("\nTraining RandomForest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    print(f"  Train accuracy: {train_acc:.3f}")

    # Optional val evaluation
    val_acc = None
    val_nearest_acc = None
    val_data = _load_val_data(args, stats, config)
    if val_data is not None:
        _, val_oracle, X_val = val_data
        centroid_array = np.array(presets)
        dists = np.linalg.norm(val_oracle[:, None, :] - centroid_array[None, :, :], axis=2)
        y_val_nearest = dists.argmin(axis=1)

        y_val_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val_nearest, y_val_pred)
        print(f"  Val classifier accuracy (vs nearest-preset): {val_acc:.3f}")
        print(f"  Random baseline: {1.0 / len(presets):.3f}")

        pred_centroids = centroid_array[y_val_pred]
        nearest_centroids = centroid_array[y_val_nearest]
        pred_mae = np.abs(val_oracle - pred_centroids).mean()
        nearest_mae = np.abs(val_oracle - nearest_centroids).mean()
        val_nearest_acc = float(val_acc)
        print(f"  Predicted preset control MAE: {pred_mae:.4f}")
        print(f"  Nearest preset control MAE (lower bound): {nearest_mae:.4f}")

    # Save outputs
    presets_data = {
        "presets": presets,
        "cluster_sizes": cluster_sizes,
        "n_clusters": len(presets),
        "hdbscan_params": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
        },
        "used_fallback_kmeans": used_fallback,
    }
    with open(output_dir / "presets.json", "w") as f:
        json.dump(presets_data, f, indent=2)

    joblib.dump(clf, output_dir / "classifier.joblib")

    report = {
        "method": "classify",
        "n_train_windows": len(train_windows),
        "n_clusters": len(presets),
        "noise_ratio_before_reassign": float(noise_ratio),
        "used_fallback_kmeans": used_fallback,
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc) if val_acc is not None else None,
        "val_nearest_accuracy": val_nearest_acc,
        "presets": presets,
        "cluster_sizes": cluster_sizes,
    }
    with open(output_dir / "preset_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved to {output_dir}:")
    print(f"  presets.json, classifier.joblib, preset_report.json")


def _run_regress(args, output_dir: Path, stats: DatasetStats) -> None:
    """Direct regression: 21-dim features → [expressiveness, dynamics_center]."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    import joblib

    train_windows, oracle_controls, X_train, config = _load_train_data(args, stats)

    print("\nTraining RandomForest regressor...")
    regressor = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    regressor.fit(X_train, oracle_controls)

    train_pred = np.clip(regressor.predict(X_train), 0.0, 1.0)
    train_mae = mean_absolute_error(oracle_controls, train_pred)
    train_mae_per_dim = np.abs(oracle_controls - train_pred).mean(axis=0).tolist()
    print(f"  Train control MAE: {train_mae:.4f}")
    print(f"    expressiveness: {train_mae_per_dim[0]:.4f}, dynamics_center: {train_mae_per_dim[1]:.4f}")

    # Optional val evaluation
    val_mae = None
    val_mae_per_dim = None
    val_data = _load_val_data(args, stats, config)
    if val_data is not None:
        _, val_oracle, X_val = val_data
        val_pred = np.clip(regressor.predict(X_val), 0.0, 1.0)
        val_mae = float(mean_absolute_error(val_oracle, val_pred))
        val_mae_per_dim = np.abs(val_oracle - val_pred).mean(axis=0).tolist()
        print(f"\n  Val control MAE: {val_mae:.4f}")
        print(f"    expressiveness: {val_mae_per_dim[0]:.4f}, dynamics_center: {val_mae_per_dim[1]:.4f}")

    # Save outputs
    joblib.dump(regressor, output_dir / "control_regressor.joblib")

    report = {
        "method": "regress",
        "n_train_windows": len(train_windows),
        "train_control_mae": float(train_mae),
        "train_control_mae_per_dim": train_mae_per_dim,
        "val_control_mae": val_mae,
        "val_control_mae_per_dim": val_mae_per_dim,
    }
    with open(output_dir / "regression_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved to {output_dir}:")
    print(f"  control_regressor.joblib, regression_report.json")


def _run_sweep(args, output_dir: Path, stats: DatasetStats) -> None:
    """KMeans K sweep (K=3,5,7,10) with automatic best-K selection."""
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib

    train_windows, oracle_controls, X_train, config = _load_train_data(args, stats)

    val_data = _load_val_data(args, stats, config)

    k_values = [3, 5, 7, 10]
    sweep_results = []

    for k in k_values:
        print(f"\n{'='*50}")
        print(f"KMeans K={k}")
        print(f"{'='*50}")

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(oracle_controls)

        # Compute centroids and cluster sizes
        presets = []
        cluster_sizes = []
        for i in range(k):
            mask = labels == i
            presets.append(oracle_controls[mask].mean(axis=0).tolist())
            cluster_sizes.append(int(mask.sum()))

        max_cluster_ratio = max(cluster_sizes) / len(labels)
        print(f"  Cluster sizes: {cluster_sizes}")
        print(f"  Max cluster ratio: {max_cluster_ratio:.1%}")

        # Nearest-preset oracle MAE (ceiling for this K)
        centroid_array = np.array(presets)
        dists = np.linalg.norm(oracle_controls[:, None, :] - centroid_array[None, :, :], axis=2)
        nearest_labels = dists.argmin(axis=1)
        nearest_centroids = centroid_array[nearest_labels]
        nearest_mae = float(np.abs(oracle_controls - nearest_centroids).mean())
        print(f"  Nearest-preset control MAE (ceiling): {nearest_mae:.4f}")

        # Train classifier
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, labels)
        train_acc = float(accuracy_score(labels, clf.predict(X_train)))
        print(f"  Train accuracy: {train_acc:.3f}")

        val_acc = None
        val_pred_mae = None
        if val_data is not None:
            _, val_oracle, X_val = val_data
            # Val nearest-preset ground truth
            val_dists = np.linalg.norm(val_oracle[:, None, :] - centroid_array[None, :, :], axis=2)
            y_val_nearest = val_dists.argmin(axis=1)

            y_val_pred = clf.predict(X_val)
            val_acc = float(accuracy_score(y_val_nearest, y_val_pred))
            val_pred_centroids = centroid_array[y_val_pred]
            val_pred_mae = float(np.abs(val_oracle - val_pred_centroids).mean())
            print(f"  Val accuracy: {val_acc:.3f} (random: {1.0/k:.3f})")
            print(f"  Val predicted-preset control MAE: {val_pred_mae:.4f}")

        result = {
            "k": k,
            "presets": presets,
            "cluster_sizes": cluster_sizes,
            "max_cluster_ratio": max_cluster_ratio,
            "nearest_preset_mae": nearest_mae,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "val_pred_mae": val_pred_mae,
            "_clf": clf,
            "_labels": labels,
        }
        sweep_results.append(result)

    # Select best K: minimize val_pred_mae if available, else nearest_preset_mae
    if val_data is not None:
        best = min(sweep_results, key=lambda r: r["val_pred_mae"])
        criterion = "val_pred_mae"
    else:
        best = min(sweep_results, key=lambda r: r["nearest_preset_mae"])
        criterion = "nearest_preset_mae"

    best_k = best["k"]
    print(f"\n{'='*50}")
    print(f"Best K={best_k} (by {criterion}={best[criterion]:.4f})")
    print(f"{'='*50}")

    # Save best K's outputs
    presets_data = {
        "presets": best["presets"],
        "cluster_sizes": best["cluster_sizes"],
        "n_clusters": best_k,
        "method": "sweep_kmeans",
        "best_k_criterion": criterion,
    }
    with open(output_dir / "presets.json", "w") as f:
        json.dump(presets_data, f, indent=2)

    joblib.dump(best["_clf"], output_dir / "classifier.joblib")

    # Build sweep report (without internal objects)
    sweep_report = []
    for r in sweep_results:
        sweep_report.append({k: v for k, v in r.items() if not k.startswith("_")})

    report = {
        "method": "sweep",
        "n_train_windows": len(train_windows),
        "k_values": k_values,
        "best_k": best_k,
        "best_k_criterion": criterion,
        "sweep_results": sweep_report,
    }
    with open(output_dir / "sweep_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved to {output_dir}:")
    print(f"  presets.json, classifier.joblib, sweep_report.json")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = _load_stats(args.stats)

    if args.method == "classify":
        _run_classify(args, output_dir, stats)
    elif args.method == "regress":
        _run_regress(args, output_dir, stats)
    elif args.method == "sweep":
        _run_sweep(args, output_dir, stats)
    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()
