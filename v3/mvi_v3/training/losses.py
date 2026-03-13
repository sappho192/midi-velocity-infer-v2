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
