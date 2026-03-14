"""Multi-alternative generation evaluation for Phase 5b (stochastic head).

Generates multiple velocity alternatives per piece and measures diversity.

Usage:
    uv run python scripts/eval_alternatives.py \
        --data-dir /path/to/test/csv \
        --checkpoint /path/to/best.pt \
        --stats /path/to/stats.json \
        --output-dir /path/to/alternatives_results \
        --n-alternatives 5
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
    parser = argparse.ArgumentParser(description="Multi-alternative velocity generation evaluation.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--n-alternatives", type=int, default=5)
    parser.add_argument("--surprise-values", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="Surprise values to evaluate")
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args()


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

    if ckpt_config.get("head_type") != "stochastic":
        print(f"ERROR: This script requires head_type='stochastic', got '{ckpt_config.get('head_type')}'")
        return

    config = BaselineConfig(
        time_scale=args.time_scale,
        head_type="stochastic",
        enable_controls=ckpt_config.get("enable_controls", False),
        control_dims=ckpt_config.get("control_dims", 3),
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
    if config.enable_controls and stats.oracle_mins:
        normalize_oracle_controls(windows, stats.oracle_mins, stats.oracle_maxs)
    loader = DataLoader(
        WindowDataset(windows, config=config, training=False),
        batch_size=config.batch_size, shuffle=False,
    )

    results = {}

    for surprise_val in args.surprise_values:
        print(f"\n--- Surprise = {surprise_val:.2f}, generating {args.n_alternatives} alternatives ---")

        # Collect hidden states from the transformer (before head)
        # We need to run the full model to get hidden states for sampling
        all_alternatives: list[list[np.ndarray]] = [[] for _ in range(args.n_alternatives)]

        with torch.no_grad():
            for batch in loader:
                pitch = batch["pitch"].to(device)
                register_bucket = batch["register_bucket"].to(device)
                continuous = batch["continuous"].to(device)
                padding_mask = batch["padding_mask"].to(device)

                # Build control params with specified surprise
                control_params = None
                if config.enable_controls:
                    if "oracle_controls" in batch:
                        oracle = batch["oracle_controls"].to(device)
                    else:
                        oracle = torch.full((pitch.shape[0], 2), 0.5, device=device)

                    if config.control_dims >= 3:
                        surprise_col = torch.full((pitch.shape[0], 1), surprise_val, device=device)
                        control_params = torch.cat([oracle, surprise_col], dim=-1)
                    else:
                        control_params = oracle

                # Forward through embedding + transformer to get hidden states
                x = model.embedding(pitch, register_bucket, continuous)
                if model.embedding_dropout is not None:
                    x = model.embedding_dropout(x)
                if model.enable_controls:
                    x = x + model.control_embedding(control_params, x.shape[0])
                bias = model.position_bias(x.shape[1], x.device)
                for layer in model.layers:
                    x = layer(x, padding_mask, bias)
                hidden = model.output_norm(x)

                # Sample alternatives
                surprise_tensor = torch.full((pitch.shape[0], 1), surprise_val, device=device)
                samples = model.head.sample(
                    hidden, n_alternatives=args.n_alternatives, surprise=surprise_tensor,
                )  # [n_alternatives, batch, seq_len]

                for alt_idx in range(args.n_alternatives):
                    all_alternatives[alt_idx].extend(samples[alt_idx].cpu().numpy())

        # Reconstruct each alternative
        reconstructed_alts = []
        for alt_idx in range(args.n_alternatives):
            recon = reconstruct_center_priority(windows, all_alternatives[alt_idx])
            reconstructed_alts.append(recon)

        # Measure diversity: pairwise L1 distance between alternatives
        piece_ids = list(reconstructed_alts[0].keys())
        pairwise_distances = []
        per_piece_diversity = {}

        for piece_id in piece_ids:
            alt_arrays = []
            for alt in reconstructed_alts:
                if piece_id in alt:
                    arr = np.array([denormalize_velocity(v, stats) for v in alt[piece_id]])
                    alt_arrays.append(arr)

            if len(alt_arrays) < 2:
                continue

            # Pairwise L1
            dists = []
            for i in range(len(alt_arrays)):
                for j in range(i + 1, len(alt_arrays)):
                    dists.append(float(np.abs(alt_arrays[i] - alt_arrays[j]).mean()))
            avg_dist = np.mean(dists)
            pairwise_distances.append(avg_dist)

            # Std across alternatives per note
            stacked = np.stack(alt_arrays)  # [n_alt, n_notes]
            note_std = stacked.std(axis=0).mean()
            per_piece_diversity[piece_id] = {
                "avg_pairwise_l1": float(avg_dist),
                "avg_note_std": float(note_std),
            }

        overall_diversity = float(np.mean(pairwise_distances)) if pairwise_distances else 0.0
        overall_note_std = float(np.mean([d["avg_note_std"] for d in per_piece_diversity.values()])) if per_piece_diversity else 0.0

        print(f"  Avg pairwise L1: {overall_diversity:.2f}")
        print(f"  Avg per-note std: {overall_note_std:.2f}")

        results[f"surprise_{surprise_val:.2f}"] = {
            "surprise": surprise_val,
            "n_alternatives": args.n_alternatives,
            "avg_pairwise_l1": overall_diversity,
            "avg_note_std": overall_note_std,
            "per_piece": per_piece_diversity,
        }

    # Summary
    print(f"\n{'='*60}")
    print("Diversity vs Surprise Summary")
    print(f"{'='*60}")
    for key in sorted(results.keys()):
        r = results[key]
        print(f"  surprise={r['surprise']:.2f}: pairwise_L1={r['avg_pairwise_l1']:.2f}, "
              f"note_std={r['avg_note_std']:.2f}")

    save_json(output_dir / "alternatives.json", results)
    print(f"\nResults saved to {output_dir / 'alternatives.json'}")


if __name__ == "__main__":
    main()
