from __future__ import annotations

import torch
from torch.utils.data import Dataset

from mvi_v3.config import BaselineConfig

from .augmentation import tempo_perturbation, velocity_jitter
from .events import WindowRecord


# Indices of timing features in the continuous feature vector
# (duration_sec=0, ioi_next_sec=1)
_TIMING_FEATURE_INDICES = [0, 1]


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        windows: list[WindowRecord],
        config: BaselineConfig | None = None,
        training: bool = False,
    ) -> None:
        self.windows = windows
        self.config = config or BaselineConfig()
        self.training = training

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        continuous = torch.as_tensor(window.continuous, dtype=torch.float32)
        target = torch.as_tensor(window.target_velocity, dtype=torch.float32)
        padding_mask = torch.as_tensor(window.padding_mask, dtype=torch.bool)

        # Data augmentation (training only)
        if self.training:
            if self.config.augment_velocity_jitter > 0:
                target = velocity_jitter(target, padding_mask, self.config.augment_velocity_jitter)
            if self.config.augment_tempo_range != (1.0, 1.0):
                continuous = tempo_perturbation(
                    continuous, self.config.augment_tempo_range, _TIMING_FEATURE_INDICES,
                )

        result: dict[str, torch.Tensor] = {
            "pitch": torch.as_tensor(window.pitch, dtype=torch.long),
            "register_bucket": torch.as_tensor(window.register_bucket, dtype=torch.long),
            "continuous": continuous,
            "target_velocity": target,
            "padding_mask": padding_mask,
        }

        # Oracle controls
        if window.oracle_controls is not None:
            result["oracle_controls"] = torch.as_tensor(window.oracle_controls, dtype=torch.float32)

        # Masked attribute regularization (Phase 3)
        if self.training and self.config.mask_ratio > 0:
            seq_len = continuous.shape[0]
            n_valid = int((~padding_mask).sum().item())
            n_mask = max(1, int(n_valid * self.config.mask_ratio))
            # Random mask indices among valid positions
            valid_indices = (~padding_mask).nonzero(as_tuple=True)[0]
            perm = torch.randperm(n_valid)[:n_mask]
            mask_indices = valid_indices[perm]
            # Store original values before masking
            original_continuous = continuous.clone()
            # Zero out all continuous features for masked positions
            continuous = continuous.clone()
            continuous[mask_indices] = 0.0
            result["continuous"] = continuous
            result["original_continuous"] = original_continuous
            # Boolean mask: True at masked positions
            attr_mask = torch.zeros(seq_len, dtype=torch.bool)
            attr_mask[mask_indices] = True
            result["attr_mask"] = attr_mask

        return result
