from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mvi_v3.config import BaselineConfig
from mvi_v3.data.datasets import WindowDataset
from mvi_v3.data.features import add_derived_features, sort_and_reindex
from mvi_v3.data.ingest import load_piece_directory
from mvi_v3.data.normalize import fit_dataset_stats
from mvi_v3.data.windowing import build_windows, fit_oracle_stats, normalize_oracle_controls
from mvi_v3.io.artifacts import save_json
from mvi_v3.models.transformer import TransformerVelocityModel
from mvi_v3.training.checkpointing import (
    load_full_checkpoint,
    restore_rng_states,
    save_full_checkpoint,
)
from mvi_v3.training.ema import ModelEMA
from mvi_v3.training.engine import run_epoch
from mvi_v3.training.monitoring import TrainingMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the v3 supervised baseline.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--velocity-weight-beta", type=float, default=0.0,
                        help="V-shaped loss weighting strength (0=off, 3=He2025 default)")
    parser.add_argument("--head-type", type=str, default="regression",
                        choices=["regression", "classification", "stochastic"],
                        help="Output head type")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing for classification head")
    # Phase 5a: Controllable inference
    parser.add_argument("--enable-controls", action="store_true",
                        help="Enable oracle-conditioned controllable inference")
    parser.add_argument("--control-dims", type=int, default=2,
                        help="Number of control dimensions (2=oracle, 3=oracle+surprise)")
    # Regularization
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer dropout rate")
    parser.add_argument("--embedding-dropout", type=float, default=0.0,
                        help="Dropout after note embedding (0=off)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay")
    # Data augmentation
    parser.add_argument("--augment-velocity-jitter", type=float, default=0.0,
                        help="Velocity jitter in raw 0-127 scale (0=off)")
    parser.add_argument("--augment-tempo-min", type=float, default=1.0,
                        help="Min tempo scaling factor")
    parser.add_argument("--augment-tempo-max", type=float, default=1.0,
                        help="Max tempo scaling factor")
    # Phase 3: Masked attribute regularization
    parser.add_argument("--mask-ratio", type=float, default=0.0,
                        help="Fraction of notes to mask for aux reconstruction (0=off)")
    parser.add_argument("--aux-loss-weight", type=float, default=0.1,
                        help="Weight of auxiliary reconstruction loss")
    return parser.parse_args()


def _process_single_piece(piece: list, config: BaselineConfig) -> list:
    return add_derived_features(sort_and_reindex(piece), config)


def prepare_pieces(path: str, config: BaselineConfig) -> list[list]:
    pieces = load_piece_directory(path, time_scale=config.time_scale)
    if not pieces:
        return []
    workers = min(len(pieces), os.cpu_count() or 1)
    fn = partial(_process_single_piece, config=config)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        prepared = list(pool.map(fn, pieces))
    return [p for p in prepared if p]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
) -> LambdaLR:
    """Linear warmup followed by cosine decay."""
    warmup_steps = int(total_steps * warmup_fraction)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BaselineConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        time_scale=args.time_scale,
        patience=args.patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        velocity_weight_beta=args.velocity_weight_beta,
        head_type=args.head_type,
        label_smoothing=args.label_smoothing,
        enable_controls=args.enable_controls,
        control_dims=args.control_dims,
        dropout=args.dropout,
        embedding_dropout=args.embedding_dropout,
        weight_decay=args.weight_decay,
        augment_velocity_jitter=args.augment_velocity_jitter,
        augment_tempo_range=(args.augment_tempo_min, args.augment_tempo_max),
        mask_ratio=args.mask_ratio,
        aux_loss_weight=args.aux_loss_weight,
    )

    # Data preparation
    print("[1/5] Loading train pieces...")
    train_pieces = prepare_pieces(args.train_dir, config)
    print(f"       {len(train_pieces)} train pieces loaded")
    print("[2/5] Loading val pieces...")
    val_pieces = prepare_pieces(args.val_dir, config)
    print(f"       {len(val_pieces)} val pieces loaded")
    print("[3/5] Computing dataset stats and building windows...")
    stats = fit_dataset_stats(train_pieces, config)
    train_windows = build_windows(train_pieces, stats, config)
    val_windows = build_windows(val_pieces, stats, config)

    # Fit and apply oracle control normalization
    if config.enable_controls:
        oracle_mins, oracle_maxs = fit_oracle_stats(train_windows)
        stats.oracle_mins = oracle_mins
        stats.oracle_maxs = oracle_maxs
        normalize_oracle_controls(train_windows, oracle_mins, oracle_maxs)
        normalize_oracle_controls(val_windows, oracle_mins, oracle_maxs)
        print(f"       Oracle controls: {len(oracle_mins)} dims, "
              f"mins={[f'{v:.2f}' for v in oracle_mins]}, "
              f"maxs={[f'{v:.2f}' for v in oracle_maxs]}")

    print(f"       {len(train_windows)} train windows, {len(val_windows)} val windows")

    train_loader = DataLoader(
        WindowDataset(train_windows, config=config, training=True),
        batch_size=config.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        WindowDataset(val_windows, config=config, training=False),
        batch_size=config.batch_size, shuffle=False,
    )

    # Model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[4/5] Building model on {device}...")
    model = TransformerVelocityModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # LR scheduler
    steps_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * config.epochs
    scheduler = build_scheduler(optimizer, total_steps, config.warmup_fraction)

    # EMA
    ema: ModelEMA | None = None
    if config.ema_enabled:
        ema = ModelEMA(model, decay=config.ema_decay)

    # Monitoring
    monitor = TrainingMonitor(output_dir)

    # Resume state
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    history: list[dict[str, float]] = []

    if args.resume_from:
        ckpt = load_full_checkpoint(args.resume_from)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_val_loss = ckpt["best_val_metric"]
        best_epoch = ckpt["best_epoch"]
        history = ckpt.get("history", [])
        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        if "rng_states" in ckpt:
            restore_rng_states(ckpt["rng_states"])
        # Recompute early stop counter from history
        for entry in reversed(history):
            if entry.get("val_loss", float("inf")) <= best_val_loss:
                break
            early_stop_counter += 1
        print(f"Resumed from {args.resume_from} at epoch {start_epoch}")

    # Print config summary
    extras = []
    if config.enable_controls:
        extras.append(f"controls={config.control_dims}d")
    if config.mask_ratio > 0:
        extras.append(f"mask={config.mask_ratio:.0%}")
    if config.augment_velocity_jitter > 0:
        extras.append(f"vel_jitter=±{config.augment_velocity_jitter}")
    if config.embedding_dropout > 0:
        extras.append(f"emb_drop={config.embedding_dropout}")
    extra_str = f" [{', '.join(extras)}]" if extras else ""

    # Training loop
    print(f"[5/5] Starting training ({config.epochs} epochs, {total_steps} total steps, "
          f"EMA={'on' if ema else 'off'}{extra_str})...")
    for epoch in range(start_epoch, config.epochs):
        epoch_num = epoch + 1
        epoch_start = time.monotonic()

        monitor.on_epoch_start(epoch_num, config.epochs)

        train_loss, steps = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scheduler=scheduler,
            max_grad_norm=config.max_grad_norm,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            huber_delta=config.huber_delta,
            velocity_weight_beta=config.velocity_weight_beta,
            ema=ema,
            head_type=config.head_type,
            num_velocity_bins=config.num_velocity_bins,
            label_smoothing=config.label_smoothing,
            enable_controls=config.enable_controls,
            mask_ratio=config.mask_ratio,
            aux_loss_weight=config.aux_loss_weight,
        )
        global_step += steps

        val_loss, _ = run_epoch(
            model, val_loader, None, device,
            huber_delta=config.huber_delta,
            velocity_weight_beta=config.velocity_weight_beta,
            head_type=config.head_type,
            num_velocity_bins=config.num_velocity_bins,
            label_smoothing=config.label_smoothing,
            enable_controls=config.enable_controls,
        )

        epoch_duration = time.monotonic() - epoch_start
        remaining_epochs = config.epochs - epoch_num
        eta_sec = epoch_duration * remaining_epochs

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_num
            early_stop_counter = 0
            save_full_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_num,
                global_step=global_step,
                best_val_metric=best_val_loss,
                best_epoch=best_epoch,
                history=history,
                config=config.to_dict(),
                ema_state_dict=ema.state_dict() if ema is not None else None,
            )
            monitor.on_checkpoint_saved(str(output_dir / "best.pt"), reason="best")
        else:
            early_stop_counter += 1

        # Record history
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else config.learning_rate
        history.append({
            "epoch": epoch_num,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
        })

        # Monitor
        monitor.on_epoch_end(
            epoch=epoch_num,
            total_epochs=config.epochs,
            global_step=global_step,
            total_steps_estimate=total_steps,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            learning_rate=current_lr,
            early_stop_counter=early_stop_counter,
            patience=config.patience,
            epoch_duration_sec=epoch_duration,
            eta_sec=eta_sec,
        )

        # Save latest checkpoint every epoch
        save_full_checkpoint(
            output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch_num,
            global_step=global_step,
            best_val_metric=best_val_loss,
            best_epoch=best_epoch,
            history=history,
            config=config.to_dict(),
            ema_state_dict=ema.state_dict() if ema is not None else None,
        )
        monitor.on_checkpoint_saved(str(output_dir / "latest.pt"), reason="periodic")

        # Early stopping
        if early_stop_counter >= config.patience:
            monitor.on_early_stop(epoch_num, best_epoch, best_val_loss)
            break
    else:
        monitor.on_training_complete(config.epochs, best_epoch, best_val_loss)

    # Save final artifacts
    save_json(output_dir / "config.json", config.to_dict())
    stats_payload = {
        "feature_means": stats.feature_means,
        "feature_stds": stats.feature_stds,
        "velocity_min": stats.velocity_min,
        "velocity_max": stats.velocity_max,
    }
    if config.enable_controls:
        stats_payload["oracle_mins"] = stats.oracle_mins
        stats_payload["oracle_maxs"] = stats.oracle_maxs
    save_json(output_dir / "stats.json", stats_payload)
    save_json(output_dir / "history.json", {"history": history})


if __name__ == "__main__":
    main()
