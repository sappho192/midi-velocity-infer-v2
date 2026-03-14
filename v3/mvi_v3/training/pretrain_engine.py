"""Training engine for Masked Note Modeling (MNM) SSL pretraining."""
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from .ema import ModelEMA


def run_pretrain_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    scheduler: object | None = None,
    max_grad_norm: float = 0.0,
    gradient_accumulation_steps: int = 1,
    ema: ModelEMA | None = None,
    pitch_weight: float = 1.0,
    continuous_weight: float = 1.0,
) -> tuple[float, float, float, int]:
    """Run one pretrain epoch.

    Returns:
        (total_loss, pitch_loss, continuous_loss, optimizer_steps)
    """
    training = optimizer is not None
    model.train(training)
    total_loss_sum = 0.0
    pitch_loss_sum = 0.0
    cont_loss_sum = 0.0
    total_batches = 0
    optimizer_steps = 0

    if training and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader):
        pitch = batch["pitch"].to(device)
        register_bucket = batch["register_bucket"].to(device)
        continuous = batch["continuous"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        mnm_mask = batch["mnm_mask"].to(device)
        original_pitch = batch["original_pitch"].to(device)
        original_continuous = batch["original_continuous"].to(device)

        with torch.set_grad_enabled(training):
            pitch_logits, cont_pred = model(
                pitch, register_bucket, continuous, padding_mask, mnm_mask,
            )

            # Pitch CE loss on masked positions only
            masked_pitch_logits = pitch_logits[mnm_mask]  # [n_masked, 128]
            masked_pitch_targets = original_pitch[mnm_mask]  # [n_masked]
            pitch_loss = F.cross_entropy(masked_pitch_logits, masked_pitch_targets)

            # Continuous MSE loss on masked positions only
            masked_cont_pred = cont_pred[mnm_mask]  # [n_masked, n_continuous]
            masked_cont_targets = original_continuous[mnm_mask]  # [n_masked, n_continuous]
            cont_loss = F.mse_loss(masked_cont_pred, masked_cont_targets)

            loss = pitch_weight * pitch_loss + continuous_weight * cont_loss

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

        total_loss_sum += float(loss.detach().cpu())
        pitch_loss_sum += float(pitch_loss.detach().cpu())
        cont_loss_sum += float(cont_loss.detach().cpu())
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

    n = max(total_batches, 1)
    return total_loss_sum / n, pitch_loss_sum / n, cont_loss_sum / n, optimizer_steps
