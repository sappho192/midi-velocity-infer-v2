from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def capture_rng_states() -> dict[str, object]:
    """Capture RNG states for Python, NumPy, PyTorch CPU, and CUDA."""
    states: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: dict[str, object]) -> None:
    """Restore RNG states for Python, NumPy, PyTorch CPU, and CUDA."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.random.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def save_checkpoint(path: str | Path, payload: dict[str, object]) -> None:
    """Save checkpoint atomically: write to temp file, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        torch.save(payload, tmp_path)
        Path(tmp_path).rename(path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    finally:
        import os

        os.close(fd)


def save_full_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None = None,
    epoch: int,
    global_step: int,
    best_val_metric: float,
    best_epoch: int,
    history: list[dict[str, float]],
    config: dict[str, object],
    ema_state_dict: dict[str, object] | None = None,
) -> None:
    """Save a full training checkpoint with all state needed for resume."""
    payload: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_metric": best_val_metric,
        "best_epoch": best_epoch,
        "history": history,
        "config": config,
        "rng_states": capture_rng_states(),
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if ema_state_dict is not None:
        payload["ema_state_dict"] = ema_state_dict
    save_checkpoint(path, payload)


def load_full_checkpoint(path: str | Path) -> dict[str, object]:
    """Load a full checkpoint, returning the payload dict."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


# Alias for backward compatibility with eval scripts
load_checkpoint = load_full_checkpoint
