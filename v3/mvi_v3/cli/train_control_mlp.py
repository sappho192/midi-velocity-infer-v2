"""Train a lightweight MLP to predict control parameters from window features.

Replaces the RandomForest regressor (~890MB joblib) with a tiny PyTorch
model (~23KB) that maps 21-dim hand-crafted window features to 2-dim
normalized controls (expressiveness, dynamics_center).

Can also merge a trained MLP into an existing model checkpoint for
single-file deployment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mvi_v3.cli.build_presets import _extract_features_matrix, _load_stats, _load_train_data, _load_val_data
from mvi_v3.io.artifacts import save_json
from mvi_v3.models.control_predictor import ControlPredictorMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train control predictor MLP or merge into checkpoint.")
    sub = parser.add_subparsers(dest="command", help="Sub-command")

    # Train sub-command
    train_p = sub.add_parser("train", help="Train the MLP from scratch")
    train_p.add_argument("--train-dir", required=True)
    train_p.add_argument("--val-dir", default=None)
    train_p.add_argument("--stats", required=True, help="Path to stats.json")
    train_p.add_argument("--output-dir", required=True)
    train_p.add_argument("--epochs", type=int, default=200)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--hidden-dim", type=int, default=64)
    train_p.add_argument("--batch-size", type=int, default=512)
    train_p.add_argument("--time-scale", type=float, default=1.0)

    # Merge sub-command
    merge_p = sub.add_parser("merge", help="Merge MLP into model checkpoint")
    merge_p.add_argument("--checkpoint", required=True, help="Model checkpoint (best.pt)")
    merge_p.add_argument("--mlp-checkpoint", required=True, help="MLP checkpoint (control_mlp.pt)")
    merge_p.add_argument("--output", default=None, help="Output path (default: overwrite checkpoint)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)
    return args


def train_mlp(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = _load_stats(args.stats)
    _, oracle_controls, X_train, config = _load_train_data(args, stats)

    val_data = _load_val_data(args, stats, config)
    X_val, val_oracle = None, None
    if val_data is not None:
        _, val_oracle, X_val = val_data

    # Convert to tensors
    X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(oracle_controls, dtype=torch.float32)
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    if X_val is not None:
        X_val_t = torch.as_tensor(X_val, dtype=torch.float32)
        y_val_t = torch.as_tensor(val_oracle, dtype=torch.float32)

    # Model
    input_dim = X_train.shape[1]
    output_dim = oracle_controls.shape[1]
    model = ControlPredictorMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20,
    )
    criterion = nn.MSELoss()

    print(f"Training ControlPredictorMLP: {input_dim} → {args.hidden_dim} → {output_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Train samples: {len(X_train_t)}, epochs: {args.epochs}")

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches

        # Validation
        val_str = ""
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_mae = float(torch.abs(val_pred - y_val_t).mean())
                val_mae_per_dim = [
                    float(torch.abs(val_pred[:, i] - y_val_t[:, i]).mean())
                    for i in range(output_dim)
                ]
            scheduler.step(val_mae)
            val_str = f"  val_mae={val_mae:.4f}"

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            scheduler.step(avg_loss)

        if epoch % 20 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.6f}{val_str}  lr={lr:.1e}")

    # Use best model if val was available
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final val metrics
    if X_val is not None:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            final_val_mae = float(torch.abs(val_pred - y_val_t).mean())
            final_val_mae_per_dim = [
                float(torch.abs(val_pred[:, i] - y_val_t[:, i]).mean())
                for i in range(output_dim)
            ]
        print(f"\nBest val MAE: {final_val_mae:.4f}")
        for i, m in enumerate(final_val_mae_per_dim):
            print(f"  dim {i}: {m:.4f}")
    else:
        final_val_mae = None
        final_val_mae_per_dim = None

    # Save
    mlp_config = {
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": output_dim,
    }
    save_path = output_dir / "control_mlp.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": mlp_config,
        "val_mae": final_val_mae,
        "val_mae_per_dim": final_val_mae_per_dim,
    }, save_path)

    file_size = save_path.stat().st_size
    print(f"\nSaved to {save_path} ({file_size:,} bytes)")

    save_json(output_dir / "control_mlp_report.json", {
        "config": mlp_config,
        "val_mae": final_val_mae,
        "val_mae_per_dim": final_val_mae_per_dim,
        "file_size_bytes": file_size,
        "n_parameters": sum(p.numel() for p in model.parameters()),
    })


def merge_mlp(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.checkpoint)
    mlp_path = Path(args.mlp_checkpoint)
    output_path = Path(args.output) if args.output else ckpt_path

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mlp_ckpt = torch.load(mlp_path, map_location="cpu", weights_only=False)

    ckpt["control_mlp"] = {
        "state_dict": mlp_ckpt["state_dict"],
        "config": mlp_ckpt["config"],
    }

    torch.save(ckpt, output_path)
    print(f"Merged control MLP into {output_path}")
    print(f"  MLP config: {mlp_ckpt['config']}")
    if mlp_ckpt.get("val_mae") is not None:
        print(f"  MLP val MAE: {mlp_ckpt['val_mae']:.4f}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_mlp(args)
    elif args.command == "merge":
        merge_mlp(args)


if __name__ == "__main__":
    main()
