import torch
from torch.utils.data import Dataset

from .events import WindowRecord


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, windows: list[WindowRecord]) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        return {
            "pitch": torch.as_tensor(window.pitch, dtype=torch.long),
            "register_bucket": torch.as_tensor(window.register_bucket, dtype=torch.long),
            "continuous": torch.as_tensor(window.continuous, dtype=torch.float32),
            "target_velocity": torch.as_tensor(window.target_velocity, dtype=torch.float32),
            "padding_mask": torch.as_tensor(window.padding_mask, dtype=torch.bool),
        }
