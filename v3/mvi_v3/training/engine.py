from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from .ema import ModelEMA
from .losses import gaussian_nll_loss, masked_cross_entropy_loss, masked_huber_loss


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
    velocity_weight_beta: float = 0.0,
    ema: ModelEMA | None = None,
    head_type: str = "regression",
    num_velocity_bins: int = 128,
    label_smoothing: float = 0.0,
    enable_controls: bool = False,
    mask_ratio: float = 0.0,
    aux_loss_weight: float = 0.1,
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
    use_aux = training and mask_ratio > 0 and hasattr(model, "has_aux_head") and model.has_aux_head

    if training and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader):
        pitch = batch["pitch"].to(device)
        register_bucket = batch["register_bucket"].to(device)
        continuous = batch["continuous"].to(device)
        target = batch["target_velocity"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        # Oracle controls
        control_params = None
        if enable_controls and "oracle_controls" in batch:
            control_params = batch["oracle_controls"].to(device)

        with torch.set_grad_enabled(training):
            # Forward pass
            if use_aux:
                output, aux_recon = model.forward_with_aux(
                    pitch, register_bucket, continuous, padding_mask, control_params,
                )
            else:
                output = model(pitch, register_bucket, continuous, padding_mask, control_params)

            # Main loss
            if head_type == "stochastic":
                mu, log_sigma = output
                loss = gaussian_nll_loss(mu, log_sigma, target, padding_mask)
            elif head_type == "classification":
                loss = masked_cross_entropy_loss(
                    output, target, padding_mask,
                    num_bins=num_velocity_bins,
                    label_smoothing=label_smoothing,
                    velocity_weight_beta=velocity_weight_beta,
                )
            else:
                loss = masked_huber_loss(
                    output, target, padding_mask,
                    delta=huber_delta, velocity_weight_beta=velocity_weight_beta,
                )

            # Auxiliary reconstruction loss (Phase 3)
            if use_aux and "attr_mask" in batch:
                attr_mask = batch["attr_mask"].to(device)  # [batch, seq_len] True=masked
                original = batch["original_continuous"].to(device)  # [batch, seq_len, n_feat]
                # MSE only on masked positions
                masked_positions = attr_mask  # [batch, seq_len]
                if masked_positions.any():
                    aux_loss = F.mse_loss(
                        aux_recon[masked_positions],
                        original[masked_positions],
                    )
                    loss = loss + aux_loss_weight * aux_loss

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
