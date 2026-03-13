import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mvi_v3.config import BaselineConfig
from mvi_v3.data.datasets import WindowDataset
from mvi_v3.data.features import add_derived_features, sort_and_reindex
from mvi_v3.data.ingest import load_piece_directory
from mvi_v3.data.normalize import fit_dataset_stats
from mvi_v3.data.windowing import build_windows
from mvi_v3.io.artifacts import save_json
from mvi_v3.models.transformer import TransformerVelocityModel
from mvi_v3.training.checkpointing import save_checkpoint
from mvi_v3.training.engine import run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the v3 supervised baseline.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--time-scale", type=float, default=1.0)
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

    config = BaselineConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        time_scale=args.time_scale,
    )

    train_pieces = prepare_pieces(args.train_dir, config)
    val_pieces = prepare_pieces(args.val_dir, config)
    stats = fit_dataset_stats(train_pieces, config)
    train_windows = build_windows(train_pieces, stats, config)
    val_windows = build_windows(val_pieces, stats, config)

    train_loader = DataLoader(WindowDataset(train_windows), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(val_windows), batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerVelocityModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    save_json(output_dir / "config.json", config.to_dict())
    save_json(
        output_dir / "stats.json",
        {
            "feature_means": stats.feature_means,
            "feature_stds": stats.feature_stds,
            "velocity_min": stats.velocity_min,
            "velocity_max": stats.velocity_max,
        },
    )
    save_json(output_dir / "history.json", {"history": history})
    save_checkpoint(
        output_dir / "checkpoint.pt",
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
        },
    )


if __name__ == "__main__":
    main()
