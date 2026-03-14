"""Dataset for Masked Note Modeling (MNM) SSL pretraining.

Reuses WindowRecord from the supervised pipeline. Adds MNM mask generation
and returns original pitch/continuous targets for the pretext task.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from mvi_v3.config import BaselineConfig

from .events import WindowRecord


class PretrainWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        windows: list[WindowRecord],
        config: BaselineConfig | None = None,
        mask_ratio: float = 0.15,
    ) -> None:
        self.windows = windows
        self.config = config or BaselineConfig()
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]

        pitch = torch.as_tensor(window.pitch, dtype=torch.long)
        register_bucket = torch.as_tensor(window.register_bucket, dtype=torch.long)
        continuous = torch.as_tensor(window.continuous, dtype=torch.float32)
        padding_mask = torch.as_tensor(window.padding_mask, dtype=torch.bool)

        # MNM targets: original pitch and continuous features
        original_pitch = pitch.clone()
        original_continuous = continuous.clone()

        # Generate MNM mask: mask_ratio fraction of valid (non-padding) notes
        seq_len = pitch.shape[0]
        n_valid = int((~padding_mask).sum().item())
        if n_valid == 0:
            mask_indices = torch.empty(0, dtype=torch.long)
        else:
            n_mask = max(1, int(n_valid * self.mask_ratio))
            valid_indices = (~padding_mask).nonzero(as_tuple=True)[0]
            perm = torch.randperm(n_valid)[:n_mask]
            mask_indices = valid_indices[perm]

        mnm_mask = torch.zeros(seq_len, dtype=torch.bool)
        mnm_mask[mask_indices] = True

        # Zero out inputs at masked positions (model will replace with mask_embedding)
        pitch_masked = pitch.clone()
        pitch_masked[mask_indices] = 0  # dummy pitch for masked positions

        register_masked = register_bucket.clone()
        register_masked[mask_indices] = 0

        continuous_masked = continuous.clone()
        continuous_masked[mask_indices] = 0.0

        return {
            "pitch": pitch_masked,
            "register_bucket": register_masked,
            "continuous": continuous_masked,
            "padding_mask": padding_mask,
            "original_pitch": original_pitch,
            "original_continuous": original_continuous,
            "mnm_mask": mnm_mask,
        }
