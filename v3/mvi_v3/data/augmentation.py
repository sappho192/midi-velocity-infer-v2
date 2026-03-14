"""Data augmentation functions for velocity inference training.

All functions operate on tensors and are applied per-sample in __getitem__.
"""

from __future__ import annotations

import torch


def velocity_jitter(
    target: torch.Tensor,
    padding_mask: torch.Tensor,
    jitter: float,
) -> torch.Tensor:
    """Add uniform noise to target velocity (normalized [0,1] scale).

    Args:
        target: [seq_len] normalized velocity
        padding_mask: [seq_len] True=padding
        jitter: noise range in raw 0-127 scale, converted to normalized
    """
    if jitter <= 0:
        return target
    noise_scale = jitter / 127.0
    noise = torch.empty_like(target).uniform_(-noise_scale, noise_scale)
    noise[padding_mask] = 0.0
    return (target + noise).clamp(0.0, 1.0)


def tempo_perturbation(
    continuous: torch.Tensor,
    tempo_range: tuple[float, float],
    timing_indices: list[int],
) -> torch.Tensor:
    """Scale timing-related features by a random factor.

    Args:
        continuous: [seq_len, n_features] normalized features
        tempo_range: (min_scale, max_scale) e.g. (0.8, 1.2)
        timing_indices: indices of timing features in continuous (e.g. duration_sec, ioi_next_sec)
    """
    if tempo_range[0] >= tempo_range[1] or (tempo_range[0] == 1.0 and tempo_range[1] == 1.0):
        return continuous
    scale = torch.empty(1).uniform_(tempo_range[0], tempo_range[1]).item()
    out = continuous.clone()
    for idx in timing_indices:
        out[:, idx] = out[:, idx] * scale
    return out
