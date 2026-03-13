import torch


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    valid = ~padding_mask
    diff = (prediction - target).abs()
    return diff[valid].mean()
