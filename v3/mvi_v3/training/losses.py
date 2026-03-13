import torch
import torch.nn.functional as F


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    valid = ~padding_mask
    diff = (prediction - target).abs()
    return diff[valid].mean()


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor,
    delta: float = 1.0,
    velocity_weight_beta: float = 0.0,
) -> torch.Tensor:
    valid = ~padding_mask
    loss = F.huber_loss(prediction, target, reduction="none", delta=delta)
    if velocity_weight_beta > 0:
        # V-shaped weighting: upweight extreme velocities (near 0 and 1)
        # Inspired by He et al. 2025: w = 1 + beta * |v - 0.5|
        weight = 1.0 + velocity_weight_beta * (target - 0.5).abs()
        loss = loss * weight
    return loss[valid].mean()


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor,
    num_bins: int = 128,
    label_smoothing: float = 0.0,
    velocity_weight_beta: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy loss for classification velocity head.

    Args:
        logits: [batch, seq_len, num_bins]
        target: [batch, seq_len] normalized velocity in [0, 1]
        padding_mask: [batch, seq_len] True=padding
        num_bins: number of velocity bins
        label_smoothing: label smoothing factor
        velocity_weight_beta: V-shaped weighting strength
    """
    valid = ~padding_mask
    # Quantize target to bin indices
    bin_idx = (target * (num_bins - 1)).round().long().clamp(0, num_bins - 1)
    # Flatten valid positions
    logits_flat = logits[valid]  # [N, num_bins]
    target_flat = bin_idx[valid]  # [N]
    loss = F.cross_entropy(logits_flat, target_flat, reduction="none",
                           label_smoothing=label_smoothing)
    if velocity_weight_beta > 0:
        target_valid = target[valid]
        weight = 1.0 + velocity_weight_beta * (target_valid - 0.5).abs()
        loss = loss * weight
    return loss.mean()


def gaussian_nll_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood for a Gaussian output head."""
    valid = ~padding_mask
    sigma = log_sigma.exp()
    nll = 0.5 * (((target - mu) / sigma) ** 2 + 2 * log_sigma + 1.8378770664093453)
    return nll[valid].mean()
