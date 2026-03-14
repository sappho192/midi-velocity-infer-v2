"""Control parameter sweep evaluation for Phase 5a.

Sweeps expressiveness and dynamics_center across [0, 1] on a fixed test set,
measuring how model predictions change with control values.

Usage:
    uv run python scripts/eval_control_sweep.py \
        --data-dir /path/to/test/csv \
        --checkpoint /path/to/best.pt \
        --stats /path/to/stats.json \
        --output-dir /path/to/sweep_results
"""
from __future__ import annotations

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
from mvi_v3.data.windowing import build_windows, normalize_oracle_controls
from mvi_v3.eval.reconstruct import reconstruct_center_priority
from mvi_v3.io.artifacts import load_json, save_json
from mvi_v3.models.transformer import TransformerVelocityModel
from mvi_v3.training.checkpointing import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control parameter sweep evaluation.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=11, help="Number of sweep steps")
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args()


def run_inference_with_controls(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    override_controls: torch.Tensor,
    head_type: str,
) -> list[np.ndarray]:
    """Run inference with overridden control parameters."""
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_size = batch["pitch"].shape[0]
            ctrl = override_controls.unsqueeze(0).expand(batch_size, -1).to(device)
            output = model(
                batch["pitch"].to(device),
                batch["register_bucket"].to(device),
                batch["continuous"].to(device),
                batch["padding_mask"].to(device),
                control_params=ctrl,
            )
            if head_type == "stochastic":
                pred = output[0]  # mu
            elif head_type == "classification":
                pred = model.head.to_scalar(output, mode="expectation")
            else:
                pred = output
            predictions.extend(pred.cpu().numpy())
    return predictions


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
    ckpt_config = checkpoint.get("config", {})
    config = BaselineConfig(
        time_scale=args.time_scale,
        head_type=ckpt_config.get("head_type", "regression"),
        num_velocity_bins=ckpt_config.get("num_velocity_bins", 128),
        enable_controls=True,
        control_dims=ckpt_config.get("control_dims", 2),
        embedding_dropout=ckpt_config.get("embedding_dropout", 0.0),
        mask_ratio=0.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerVelocityModel(config).to(device)
    use_ema = not args.no_ema and "ema_state_dict" in checkpoint
    if use_ema:
        model.load_state_dict(checkpoint["ema_state_dict"]["shadow"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test data
    pieces = load_piece_directory(args.data_dir, time_scale=config.time_scale)
    pieces = [add_derived_features(sort_and_reindex(p), config) for p in pieces if p]
    windows = build_windows(pieces, stats, config)
    if stats.oracle_mins:
        normalize_oracle_controls(windows, stats.oracle_mins, stats.oracle_maxs)
    loader = DataLoader(
        WindowDataset(windows, config=config, training=False),
        batch_size=config.batch_size, shuffle=False,
    )

    n_controls = config.control_dims
    sweep_values = np.linspace(0.0, 1.0, args.n_steps)
    control_names = ["expressiveness", "dynamics_center", "surprise"][:n_controls]

    results = {}

    # Oracle baseline: use actual oracle controls
    print("Running oracle baseline...")
    oracle_preds = []
    with torch.no_grad():
        for batch in loader:
            fwd_kwargs = {
                "pitch": batch["pitch"].to(device),
                "register_bucket": batch["register_bucket"].to(device),
                "continuous": batch["continuous"].to(device),
                "padding_mask": batch["padding_mask"].to(device),
            }
            if "oracle_controls" in batch:
                fwd_kwargs["control_params"] = batch["oracle_controls"].to(device)
            output = model(**fwd_kwargs)
            if config.head_type == "stochastic":
                pred = output[0]
            elif config.head_type == "classification":
                pred = model.head.to_scalar(output, mode="expectation")
            else:
                pred = output
            oracle_preds.extend(pred.cpu().numpy())

    oracle_recon = reconstruct_center_priority(windows, oracle_preds)
    oracle_all = []
    for values in oracle_recon.values():
        oracle_all.extend([denormalize_velocity(v, stats) for v in values])
    oracle_all = np.array(oracle_all)
    results["oracle"] = {
        "pred_mean": float(oracle_all.mean()),
        "pred_std": float(oracle_all.std()),
    }
    print(f"  Oracle: pred_mean={oracle_all.mean():.2f}, pred_std={oracle_all.std():.2f}")

    # Sweep each control dimension
    for dim_idx, dim_name in enumerate(control_names):
        print(f"\nSweeping {dim_name} (dim {dim_idx})...")
        dim_results = []

        for val in sweep_values:
            # Set all controls to 0.5 (neutral), then override the swept dimension
            controls = torch.full((n_controls,), 0.5)
            controls[dim_idx] = val

            preds = run_inference_with_controls(model, loader, device, controls, config.head_type)
            recon = reconstruct_center_priority(windows, preds)

            all_preds = []
            for piece_values in recon.values():
                all_preds.extend([denormalize_velocity(v, stats) for v in piece_values])
            all_preds = np.array(all_preds)

            entry = {
                "control_value": float(val),
                "pred_mean": float(all_preds.mean()),
                "pred_std": float(all_preds.std()),
                "pred_min": float(all_preds.min()),
                "pred_max": float(all_preds.max()),
            }
            dim_results.append(entry)
            print(f"  {dim_name}={val:.2f}: pred_mean={all_preds.mean():.2f}, pred_std={all_preds.std():.2f}")

        results[dim_name] = dim_results

    # Monotonicity check
    print(f"\n{'='*60}")
    print("Monotonicity Analysis")
    print(f"{'='*60}")
    for dim_name in control_names:
        entries = results[dim_name]
        if dim_name == "expressiveness":
            values = [e["pred_std"] for e in entries]
            metric = "pred_std"
        elif dim_name == "dynamics_center":
            values = [e["pred_mean"] for e in entries]
            metric = "pred_mean"
        else:
            values = [e["pred_std"] for e in entries]
            metric = "pred_std"

        diffs = np.diff(values)
        monotone_increasing = all(d >= 0 for d in diffs)
        correlation = np.corrcoef(sweep_values, values)[0, 1]
        print(f"  {dim_name} → {metric}: monotone={'yes' if monotone_increasing else 'no'}, "
              f"correlation={correlation:.3f}, range=[{min(values):.2f}, {max(values):.2f}]")

    save_json(output_dir / "control_sweep.json", results)
    print(f"\nResults saved to {output_dir / 'control_sweep.json'}")


if __name__ == "__main__":
    main()
