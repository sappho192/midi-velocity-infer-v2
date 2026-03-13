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
from mvi_v3.data.windowing import build_windows
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
    )
    checkpoint = load_checkpoint(args.checkpoint)
    config = BaselineConfig(time_scale=args.time_scale)
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
    loader = DataLoader(WindowDataset(windows), batch_size=config.batch_size, shuffle=False)

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["pitch"].to(device),
                batch["register_bucket"].to(device),
                batch["continuous"].to(device),
                batch["padding_mask"].to(device),
            )
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
