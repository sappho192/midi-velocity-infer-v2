from pathlib import Path

import torch


def save_checkpoint(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path) -> dict[str, object]:
    return torch.load(Path(path), map_location="cpu")
