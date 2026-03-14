import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvi_v3.config import BaselineConfig
from mvi_v3.data.datasets import WindowDataset
from mvi_v3.data.events import DatasetStats
from mvi_v3.data.features import add_derived_features, sort_and_reindex
from mvi_v3.data.ingest import load_piece_directory
from mvi_v3.data.normalize import denormalize_velocity
from mvi_v3.data.preset_features import extract_window_features
from mvi_v3.data.windowing import build_windows, normalize_oracle_controls
from mvi_v3.eval.metrics import aggregate_metrics, compute_piece_metrics
from mvi_v3.eval.reconstruct import reconstruct_center_priority
from mvi_v3.io.artifacts import load_json, save_json
from mvi_v3.models.transformer import TransformerVelocityModel
from mvi_v3.training.checkpointing import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the v3 supervised baseline.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--no-ema", action="store_true", help="Use training weights instead of EMA weights")
    parser.add_argument("--decode-mode", type=str, default="expectation",
                        choices=["expectation", "argmax"],
                        help="Decoding mode for classification head")
    parser.add_argument("--control-mode", type=str, default=None,
                        choices=["oracle", "default", "preset", "regression", "soft_preset"],
                        help="Control mode: oracle/default/preset/regression/soft_preset")
    parser.add_argument("--preset-dir", type=str, default=None,
                        help="Directory containing presets/classifier/regressor artifacts")
    return parser.parse_args()


def prepare_pieces(path: str, config: BaselineConfig) -> list[list]:
    pieces = load_piece_directory(path, time_scale=config.time_scale)
    prepared = []
    for piece in pieces:
        prepared.append(add_derived_features(sort_and_reindex(piece), config))
    return prepared


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_payload = load_json(args.stats)
    stats = DatasetStats(
        feature_means={k: float(v) for k, v in stats_payload["feature_means"].items()},
        feature_stds={k: float(v) for k, v in stats_payload["feature_stds"].items()},
        velocity_min=float(stats_payload["velocity_min"]),
        velocity_max=float(stats_payload["velocity_max"]),
        oracle_mins=stats_payload.get("oracle_mins", []),
        oracle_maxs=stats_payload.get("oracle_maxs", []),
    )
    checkpoint = load_checkpoint(args.checkpoint)
    # Restore model config from checkpoint if available
    ckpt_config = checkpoint.get("config", {})
    enable_controls = ckpt_config.get("enable_controls", False)
    config = BaselineConfig(
        time_scale=args.time_scale,
        head_type=ckpt_config.get("head_type", "regression"),
        num_velocity_bins=ckpt_config.get("num_velocity_bins", 128),
        enable_controls=enable_controls,
        control_dims=ckpt_config.get("control_dims", 2),
        embedding_dropout=ckpt_config.get("embedding_dropout", 0.0),
        mask_ratio=0.0,  # no masking during eval
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerVelocityModel(config).to(device)
    use_ema = not args.no_ema and "ema_state_dict" in checkpoint
    if use_ema:
        model.load_state_dict(checkpoint["ema_state_dict"]["shadow"])
        print("Using EMA weights for evaluation")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Using training weights for evaluation")
    model.eval()

    pieces = prepare_pieces(args.data_dir, config)
    windows = build_windows(pieces, stats, config)

    # Determine control mode
    control_mode = args.control_mode
    if control_mode is None:
        control_mode = "oracle" if enable_controls else None

    if enable_controls and control_mode == "oracle":
        # Normalize oracle controls for test set using training stats
        if stats.oracle_mins:
            normalize_oracle_controls(windows, stats.oracle_mins, stats.oracle_maxs)
    elif enable_controls and control_mode == "preset":
        # Load presets + classifier, predict preset for each window
        if not args.preset_dir:
            raise ValueError("--preset-dir is required for --control-mode preset")
        import joblib
        preset_dir = Path(args.preset_dir)
        preset_data = load_json(str(preset_dir / "presets.json"))
        presets = preset_data["presets"]
        clf = joblib.load(preset_dir / "classifier.joblib")
        features = np.stack([extract_window_features(w) for w in windows])
        labels = clf.predict(features)
        for w, label in zip(windows, labels):
            w.oracle_controls = np.array(presets[label], dtype=np.float32)
        print(f"  Preset mode: {len(presets)} presets, assigned to {len(windows)} windows")
    elif enable_controls and control_mode == "regression":
        # Direct regression: 21-dim features → continuous controls
        if not args.preset_dir:
            raise ValueError("--preset-dir is required for --control-mode regression")
        import joblib
        preset_dir = Path(args.preset_dir)
        regressor = joblib.load(preset_dir / "control_regressor.joblib")
        features = np.stack([extract_window_features(w) for w in windows])
        predicted = np.clip(regressor.predict(features), 0.0, 1.0)
        for w, ctrl in zip(windows, predicted):
            w.oracle_controls = ctrl.astype(np.float32)
        print(f"  Regression mode: predicted controls for {len(windows)} windows")
    elif enable_controls and control_mode == "soft_preset":
        # Soft ensemble: top-2 weighted blending from classifier probabilities
        if not args.preset_dir:
            raise ValueError("--preset-dir is required for --control-mode soft_preset")
        import joblib
        preset_dir = Path(args.preset_dir)
        preset_data = load_json(str(preset_dir / "presets.json"))
        presets = preset_data["presets"]
        clf = joblib.load(preset_dir / "classifier.joblib")
        features = np.stack([extract_window_features(w) for w in windows])
        proba = clf.predict_proba(features)  # [N, K]
        centroid_array = np.array(presets)
        for w, p in zip(windows, proba):
            top2 = np.argsort(p)[-2:][::-1]
            weights = p[top2] / p[top2].sum()
            blended = weights[0] * centroid_array[top2[0]] + weights[1] * centroid_array[top2[1]]
            w.oracle_controls = np.clip(blended, 0.0, 1.0).astype(np.float32)
        print(f"  Soft preset mode: {len(presets)} presets, blended for {len(windows)} windows")
    elif enable_controls and control_mode == "default":
        # Use [0.5, 0.5] as default control params (mid-range of normalized space)
        # Note: learned default_controls may be untrained if oracle was always provided
        default_ctrl = np.array([0.5] * config.control_dims, dtype=np.float32)
        for w in windows:
            w.oracle_controls = default_ctrl

    loader = DataLoader(
        WindowDataset(windows, config=config, training=False),
        batch_size=config.batch_size, shuffle=False,
    )

    head_type = config.head_type
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            fwd_kwargs: dict = {
                "pitch": batch["pitch"].to(device),
                "register_bucket": batch["register_bucket"].to(device),
                "continuous": batch["continuous"].to(device),
                "padding_mask": batch["padding_mask"].to(device),
            }
            if enable_controls and "oracle_controls" in batch:
                fwd_kwargs["control_params"] = batch["oracle_controls"].to(device)

            output = model(**fwd_kwargs)

            if head_type == "stochastic":
                mu, _log_sigma = output
                pred = mu  # use mean for evaluation
            elif head_type == "classification":
                pred = model.head.to_scalar(output, mode=args.decode_mode)
            else:
                pred = output
            predictions.extend(pred.cpu().numpy())

    reconstructed = reconstruct_center_priority(windows, predictions)
    targets = {
        piece[0].piece_id: [float(event.velocity or 0) for event in piece]
        for piece in pieces
        if piece
    }
    piece_metrics = {
        piece_id: compute_piece_metrics(
            [denormalize_velocity(value, stats) for value in values],
            targets[piece_id],
        )
        for piece_id, values in reconstructed.items()
    }
    agg = aggregate_metrics(piece_metrics)
    save_json(output_dir / "metrics.json", {
        "aggregate": agg,
        "per_piece": piece_metrics,
    })

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({agg['n_pieces']} pieces, {agg['n_notes']} notes)")
    if enable_controls:
        print(f"  [Control mode: {control_mode}]")
    print(f"{'='*60}")
    print(f"  MAE:        {agg['weighted_mae']:.2f}  (macro {agg['macro_mae']:.2f})")
    print(f"  MSE:        {agg['weighted_mse']:.2f}  (macro {agg['macro_mse']:.2f})")
    print(f"  SD_velo:    {agg['weighted_pred_std']:.2f}  (true: {agg['weighted_true_std']:.2f})")
    print(f"  SD_ratio:   {agg['weighted_sd_ratio']:.1%}")
    print(f"  SD_ae:      {agg['weighted_sd_ae']:.2f}")
    print(f"  CC:         {agg['weighted_cc']:.4f}")
    print(f"  Recall(10%): {agg['weighted_recall_10']:.1%}")
    print(f"  Recall(5%):  {agg['weighted_recall_5']:.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
