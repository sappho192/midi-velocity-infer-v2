"""SSL pretraining CLI using Masked Note Modeling (MNM) on GiantMIDI-Piano.

Usage:
    uv run python -m mvi_v3.cli.pretrain_ssl \
        --train-dir /path/to/GiantMIDI-PIano/csv/train \
        --val-dir /path/to/GiantMIDI-PIano/csv/validation \
        --output-dir runs/ssl_pretrain \
        --epochs 50 --mask-ratio 0.15 --batch-size 256
"""
from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mvi_v3.config import BaselineConfig
from mvi_v3.data.pretrain_dataset import PretrainWindowDataset
from mvi_v3.data.events import DatasetStats, NoteEvent, WindowRecord
from mvi_v3.data.features import add_derived_features, sort_and_reindex
from mvi_v3.data.ingest import load_piece_csv
from mvi_v3.data.normalize import fit_dataset_stats
from mvi_v3.data.windowing import build_windows
from mvi_v3.io.artifacts import save_json
from mvi_v3.models.pretrain_model import PretrainModel
from mvi_v3.training.checkpointing import save_checkpoint
from mvi_v3.training.ema import ModelEMA
from mvi_v3.training.pretrain_engine import run_pretrain_epoch

# Reuse from train_baseline
from mvi_v3.cli.train_baseline import build_scheduler


def load_pieces_and_build_windows(
    path: str,
    config: BaselineConfig,
    stats: DatasetStats | None = None,
) -> tuple[list[list[NoteEvent]] | None, list[WindowRecord]]:
    """Load pieces sequentially, optionally build windows piece-by-piece.

    If stats is None, returns (pieces, []) for stats fitting.
    If stats is provided, returns (None, windows) — pieces are freed after windowing.
    """
    csv_paths = sorted(Path(path).glob("*.csv"))
    if not csv_paths:
        return ([] if stats is None else None), []

    if stats is None:
        # First pass: need to keep pieces for stats fitting
        pieces: list[list[NoteEvent]] = []
        for i, csv_path in enumerate(csv_paths, 1):
            piece = load_piece_csv(csv_path)
            if piece:
                pieces.append(add_derived_features(sort_and_reindex(piece), config))
            if i % 500 == 0:
                print(f"       ... {i}/{len(csv_paths)} pieces loaded")
        return pieces, []
    else:
        # Second pass (or val): build windows per-piece, free piece immediately
        all_windows: list[WindowRecord] = []
        for i, csv_path in enumerate(csv_paths, 1):
            piece = load_piece_csv(csv_path)
            if piece:
                processed = add_derived_features(sort_and_reindex(piece), config)
                windows = build_windows([processed], stats, config)
                all_windows.extend(windows)
            if i % 500 == 0:
                print(f"       ... {i}/{len(csv_paths)} pieces processed ({len(all_windows)} windows)")
        return None, all_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSL pretrain with Masked Note Modeling")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embedding-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--pitch-weight", type=float, default=1.0,
                        help="Weight for pitch CE loss")
    parser.add_argument("--continuous-weight", type=float, default=1.0,
                        help="Weight for continuous MSE loss")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BaselineConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        dropout=args.dropout,
        embedding_dropout=args.embedding_dropout,
        weight_decay=args.weight_decay,
    )

    # [1/5] Load training pieces (sequential, single-process to avoid fork OOM)
    print(f"[1/5] Loading train pieces from {args.train_dir}...")
    train_pieces, _ = load_pieces_and_build_windows(args.train_dir, config, stats=None)
    print(f"       {len(train_pieces)} train pieces loaded")

    # [2/5] Compute stats, then build windows piece-by-piece and free pieces
    print("[2/5] Computing dataset stats...")
    stats = fit_dataset_stats(train_pieces, config)
    del train_pieces
    gc.collect()

    # [3/5] Build windows (re-reads CSVs, but frees each piece after windowing)
    print("[3/5] Building train windows (streaming)...")
    _, train_windows = load_pieces_and_build_windows(args.train_dir, config, stats=stats)
    print(f"       {len(train_windows)} train windows")

    print("       Building val windows (streaming)...")
    _, val_windows = load_pieces_and_build_windows(args.val_dir, config, stats=stats)
    print(f"       {len(val_windows)} val windows")

    # Save pretrain stats for reference
    stats_payload = {
        "feature_means": stats.feature_means,
        "feature_stds": stats.feature_stds,
        "velocity_min": stats.velocity_min,
        "velocity_max": stats.velocity_max,
    }
    save_json(output_dir / "pretrain_stats.json", stats_payload)

    train_loader = DataLoader(
        PretrainWindowDataset(train_windows, config=config, mask_ratio=args.mask_ratio),
        batch_size=config.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        PretrainWindowDataset(val_windows, config=config, mask_ratio=args.mask_ratio),
        batch_size=config.batch_size, shuffle=False,
    )

    # [4/5] Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[4/5] Building PretrainModel on {device}...")
    model = PretrainModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"       {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.epochs
    scheduler = build_scheduler(optimizer, total_steps, config.warmup_fraction)

    ema: ModelEMA | None = None
    if config.ema_enabled:
        ema = ModelEMA(model, decay=config.ema_decay)

    # [5/5] Training loop
    print(f"[5/5] Starting SSL pretraining ({config.epochs} epochs, "
          f"mask_ratio={args.mask_ratio}, {total_steps} total steps)...")

    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    global_step = 0
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        epoch_num = epoch + 1
        epoch_start = time.monotonic()

        train_loss, train_pitch, train_cont, steps = run_pretrain_epoch(
            model, train_loader, optimizer, device,
            scheduler=scheduler,
            max_grad_norm=config.max_grad_norm,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            ema=ema,
            pitch_weight=args.pitch_weight,
            continuous_weight=args.continuous_weight,
        )
        global_step += steps

        val_loss, val_pitch, val_cont, _ = run_pretrain_epoch(
            model, val_loader, None, device,
            pitch_weight=args.pitch_weight,
            continuous_weight=args.continuous_weight,
        )

        epoch_duration = time.monotonic() - epoch_start
        remaining = config.epochs - epoch_num
        eta_min = (epoch_duration * remaining) / 60

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else config.learning_rate

        # Track best
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_num
            early_stop_counter = 0
            improved = " *"

            # Save best backbone
            backbone_payload = {
                "backbone_state_dict": model.backbone_state_dict(),
                "config": config.to_dict(),
                "epoch": epoch_num,
                "val_loss": val_loss,
            }
            save_checkpoint(output_dir / "backbone.pt", backbone_payload)

            # Also save full pretrain checkpoint for resuming
            full_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch_num,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "config": config.to_dict(),
            }
            if ema is not None:
                full_payload["ema_state_dict"] = ema.state_dict()
            save_checkpoint(output_dir / "best.pt", full_payload)
        else:
            early_stop_counter += 1

        history.append({
            "epoch": epoch_num,
            "train_loss": train_loss,
            "train_pitch_loss": train_pitch,
            "train_cont_loss": train_cont,
            "val_loss": val_loss,
            "val_pitch_loss": val_pitch,
            "val_cont_loss": val_cont,
            "learning_rate": current_lr,
        })

        print(f"  [{epoch_num:3d}/{config.epochs}] "
              f"train={train_loss:.4f} (pitch={train_pitch:.4f} cont={train_cont:.4f}) | "
              f"val={val_loss:.4f} (pitch={val_pitch:.4f} cont={val_cont:.4f}) | "
              f"best={best_val_loss:.4f}@{best_epoch} | "
              f"lr={current_lr:.2e} | "
              f"{epoch_duration:.0f}s (eta {eta_min:.0f}m){improved}")

        # Early stopping
        if early_stop_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch_num} (best={best_val_loss:.4f}@{best_epoch})")
            break
    else:
        print(f"  Training complete ({config.epochs} epochs, best={best_val_loss:.4f}@{best_epoch})")

    # Save final artifacts
    save_json(output_dir / "config.json", config.to_dict())
    save_json(output_dir / "history.json", {"history": history})
    print(f"\nBackbone saved to {output_dir / 'backbone.pt'}")


if __name__ == "__main__":
    main()
