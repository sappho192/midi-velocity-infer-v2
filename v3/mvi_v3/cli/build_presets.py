"""Build control presets via HDBSCAN clustering + RandomForest classifier.

Pipeline:
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
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--time-scale", type=float, default=1.0)
    return parser.parse_args()


def _prepare_pieces(path: str, config: BaselineConfig) -> list[list]:
    pieces = load_piece_directory(path, time_scale=config.time_scale)
    return [add_derived_features(sort_and_reindex(piece), config) for piece in pieces]


def _extract_features_matrix(windows: list) -> np.ndarray:
    return np.stack([extract_window_features(w) for w in windows])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports for optional dependencies
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib

    # Load stats
    stats_payload = load_json(args.stats)
    stats = DatasetStats(
        feature_means={k: float(v) for k, v in stats_payload["feature_means"].items()},
        feature_stds={k: float(v) for k, v in stats_payload["feature_stds"].items()},
        velocity_min=float(stats_payload["velocity_min"]),
        velocity_max=float(stats_payload["velocity_max"]),
        oracle_mins=stats_payload.get("oracle_mins", []),
        oracle_maxs=stats_payload.get("oracle_maxs", []),
    )

    config = BaselineConfig(time_scale=args.time_scale, enable_controls=True)

    # --- Step 1: Build train windows ---
    print("Loading training pieces...")
    train_pieces = _prepare_pieces(args.train_dir, config)
    train_windows = build_windows(train_pieces, stats, config)
    print(f"  {len(train_windows)} windows from {len(train_pieces)} pieces")

    # Normalize oracle controls to [0,1]
    normalize_oracle_controls(train_windows, stats.oracle_mins, stats.oracle_maxs)

    # Collect normalized oracle controls [N, 2]
    oracle_controls = np.stack([w.oracle_controls for w in train_windows])
    print(f"  Oracle controls shape: {oracle_controls.shape}")

    # --- Step 2: HDBSCAN clustering ---
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

    # --- Fallback: if HDBSCAN gives K<=1, use KMeans ---
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

    # --- Step 3: Reassign noise to nearest centroid ---
    # Compute centroids first
    unique_labels = sorted(set(labels) - {-1})
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[label] = oracle_controls[mask].mean(axis=0)

    if n_noise > 0 and not used_fallback:
        centroid_array = np.stack([centroids[l] for l in unique_labels])
        noise_mask = labels == -1
        noise_points = oracle_controls[noise_mask]
        # Nearest centroid assignment
        dists = np.linalg.norm(noise_points[:, None, :] - centroid_array[None, :, :], axis=2)
        nearest = np.array(unique_labels)[dists.argmin(axis=1)]
        labels[noise_mask] = nearest
        print(f"  Reassigned {n_noise} noise points to nearest centroids")
        # Recompute centroids after reassignment
        for label in unique_labels:
            mask = labels == label
            centroids[label] = oracle_controls[mask].mean(axis=0)

    # --- Step 4: Build preset list ---
    # Remap labels to 0..K-1
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

    # --- Step 5: Extract features + train classifier ---
    print("\nExtracting window features...")
    X_train = _extract_features_matrix(train_windows)
    y_train = labels
    print(f"  Feature matrix: {X_train.shape}")

    print("Training RandomForest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    print(f"  Train accuracy: {train_acc:.3f}")

    # --- Step 6: Optional val evaluation ---
    val_acc = None
    val_nearest_acc = None
    if args.val_dir:
        print("\nEvaluating on validation set...")
        val_pieces = _prepare_pieces(args.val_dir, config)
        val_windows = build_windows(val_pieces, stats, config)
        normalize_oracle_controls(val_windows, stats.oracle_mins, stats.oracle_maxs)

        X_val = _extract_features_matrix(val_windows)
        val_oracle = np.stack([w.oracle_controls for w in val_windows])

        # Ground truth: nearest preset for each val window
        centroid_array = np.array(presets)
        dists = np.linalg.norm(val_oracle[:, None, :] - centroid_array[None, :, :], axis=2)
        y_val_nearest = dists.argmin(axis=1)

        # Classifier predictions
        y_val_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val_nearest, y_val_pred)
        print(f"  Val classifier accuracy (vs nearest-preset): {val_acc:.3f}")
        print(f"  Random baseline: {1.0 / len(presets):.3f}")

        # Also report MAE of predicted preset vs oracle controls
        pred_centroids = centroid_array[y_val_pred]
        nearest_centroids = centroid_array[y_val_nearest]
        pred_mae = np.abs(val_oracle - pred_centroids).mean()
        nearest_mae = np.abs(val_oracle - nearest_centroids).mean()
        val_nearest_acc = float(val_acc)
        print(f"  Predicted preset control MAE: {pred_mae:.4f}")
        print(f"  Nearest preset control MAE (lower bound): {nearest_mae:.4f}")

    # --- Step 7: Save outputs ---
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


if __name__ == "__main__":
    main()
