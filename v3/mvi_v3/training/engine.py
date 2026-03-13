from __future__ import annotations

from collections.abc import Iterable

import torch

from .ema import ModelEMA
from .losses import masked_huber_loss


def run_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    scheduler: object | None = None,
    max_grad_norm: float = 0.0,
    gradient_accumulation_steps: int = 1,
    huber_delta: float = 1.0,
    ema: ModelEMA | None = None,
) -> tuple[float, int]:
    """Run one training or validation epoch.

    Returns:
        (average_loss, steps_completed) where steps_completed counts
        optimizer steps (accounting for gradient accumulation).
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    optimizer_steps = 0

    if training and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader):
        pitch = batch["pitch"].to(device)
        register_bucket = batch["register_bucket"].to(device)
        continuous = batch["continuous"].to(device)
        target = batch["target_velocity"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.set_grad_enabled(training):
            prediction = model(pitch, register_bucket, continuous, padding_mask)
            loss = masked_huber_loss(prediction, target, padding_mask, delta=huber_delta)

            if training:
                scaled_loss = loss / gradient_accumulation_steps
                scaled_loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    if ema is not None:
                        ema.update(model)
                    if scheduler is not None and hasattr(scheduler, "step"):
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

        total_loss += float(loss.detach().cpu())
        total_batches += 1

    # Handle remaining accumulated gradients
    if training and optimizer is not None and total_batches % gradient_accumulation_steps != 0:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        if scheduler is not None and hasattr(scheduler, "step"):
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss, optimizer_steps
