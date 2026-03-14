import numpy as np
import pytest
import torch

from mvi_v3.config import BaselineConfig
from mvi_v3.data.events import WindowRecord
from mvi_v3.data.pretrain_dataset import PretrainWindowDataset
from mvi_v3.models.pretrain_model import PretrainModel
from mvi_v3.training.pretrain_engine import run_pretrain_epoch


def _window(all_padding: bool) -> WindowRecord:
    seq_len = 4
    pitch = np.array([60, 62, 64, 65], dtype=np.int64)
    register = np.array([1, 1, 2, 2], dtype=np.int64)
    continuous = np.zeros((seq_len, 6), dtype=np.float32)
    target_velocity = np.array([64, 64, 64, 64], dtype=np.int64)
    padding_mask = np.ones(seq_len, dtype=bool) if all_padding else np.zeros(seq_len, dtype=bool)
    global_note_indices = np.arange(seq_len, dtype=np.int64)
    return WindowRecord(
        piece_id="p",
        start_note_index=0,
        true_length=0 if all_padding else seq_len,
        pitch=pitch,
        register_bucket=register,
        continuous=continuous,
        target_velocity=target_velocity,
        padding_mask=padding_mask,
        global_note_indices=global_note_indices,
        oracle_controls=None,
    )


def test_pretrain_dataset_handles_all_padding_window():
    ds = PretrainWindowDataset([_window(all_padding=True)], config=BaselineConfig(), mask_ratio=0.15)
    sample = ds[0]
    assert sample["mnm_mask"].sum().item() == 0


def test_pretrain_engine_handles_zero_masked_positions_without_crash():
    cfg = BaselineConfig()
    ds = PretrainWindowDataset([_window(all_padding=True)], config=cfg, mask_ratio=0.15)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    model = PretrainModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    total, pitch, cont, steps = run_pretrain_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        gradient_accumulation_steps=1,
    )

    assert total == 0.0
    assert pitch == 0.0
    assert cont == 0.0
    assert steps == 1


def test_mask_ratio_validation():
    cfg = BaselineConfig()
    with pytest.raises(ValueError, match="mask_ratio"):
        PretrainWindowDataset([_window(all_padding=False)], config=cfg, mask_ratio=0.0)
    with pytest.raises(ValueError, match="mask_ratio"):
        PretrainWindowDataset([_window(all_padding=False)], config=cfg, mask_ratio=1.5)
