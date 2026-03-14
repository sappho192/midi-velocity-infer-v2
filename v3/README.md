# MIDI Velocity Inference v3

Transformer-based piano MIDI velocity inference system. Oracle-conditioned controllable inference and SSL pretraining achieve MAE 9.81 on MAESTROv3 test set.

## Key Results

| Mode | MAE | CC | SD_ratio | Recall(10%) |
|------|-----|----|----------|-------------|
| Oracle (ground truth controls) | 7.94 | 0.797 | 91.9% | 81.2% |
| Regression (auto controls) | 9.81 | 0.717 | 85.3% | 72.1% |
| Default [0.5, 0.5] | 14.60 | 0.617 | 96.1% | 52.0% |
| v2 baseline (Seq2Seq) | 13.87 | 0.344 | 36.1% | 52.4% |

Reference: He et al. 2025 MAESTROv3 test MAE=11.5, SD_velo=10.7.

## Architecture

- **Backbone**: 4-layer Transformer encoder (d=128, 4 heads)
- **Input features**: Pitch embedding + register bucket + 6 continuous features (onset, duration, IOI, chord_size, local_density, local_pitch_range)
- **Control conditioning**: 2D control params [expressiveness, dynamics_center] via FiLM (Feature-wise Linear Modulation)
- **Output**: Regression head (Huber loss, V-shaped weighting β=3)
- **SSL pretraining**: Masked Note Modeling on GiantMIDI-Piano (5,959 pieces) → MAESTRO fine-tune

## Project Structure

```
v3/
├── mvi_v3/
│   ├── cli/
│   │   ├── train_baseline.py     # Supervised training CLI
│   │   ├── eval_baseline.py      # Evaluation (oracle/default/regression/preset/soft_preset)
│   │   ├── build_presets.py      # Control preset builder (classify/regress/sweep)
│   │   └── pretrain_ssl.py       # SSL pretraining CLI
│   ├── data/
│   │   ├── ingest.py             # MIDI CSV loading
│   │   ├── events.py             # NoteEvent, Window, DatasetStats
│   │   ├── features.py           # Derived feature computation
│   │   ├── windowing.py          # Sliding window construction
│   │   ├── datasets.py           # PyTorch Dataset
│   │   ├── preset_features.py    # 21-dim window feature extraction
│   │   ├── pretrain_dataset.py   # MNM masking dataset
│   │   ├── normalize.py          # Velocity normalization
│   │   └── augmentation.py       # Data augmentation
│   ├── models/
│   │   ├── transformer.py        # TransformerVelocityModel
│   │   ├── embedding.py          # Pitch + register embeddings
│   │   ├── position.py           # Sinusoidal positional encoding
│   │   ├── conditioning.py       # FiLM conditioning layer
│   │   ├── heads.py              # Regression/Classification/Stochastic heads
│   │   └── pretrain_model.py     # SSL PretrainModel
│   ├── training/
│   │   ├── engine.py             # Training loop
│   │   ├── pretrain_engine.py    # SSL training loop
│   │   ├── losses.py             # Huber + V-shaped weighting + CE
│   │   ├── ema.py                # Exponential Moving Average
│   │   ├── checkpointing.py      # Checkpoint save/load
│   │   └── monitoring.py         # Training logging
│   ├── eval/
│   │   ├── metrics.py            # He2025 evaluation metrics
│   │   └── reconstruct.py        # Window → piece reconstruction
│   ├── io/
│   │   └── artifacts.py          # JSON I/O
│   └── config.py                 # BaselineConfig dataclass
├── scripts/                      # Utility scripts
│   ├── convert_giantmidi_to_csv.py
│   └── ...
├── docs/
│   ├── impl-status.md            # Full experiment log
│   ├── control-preset-v2-result.md
│   └── ...
└── pyproject.toml
```

## Usage

```bash
# Training (SSL pretrained backbone)
uv run python -m mvi_v3.cli.pretrain_ssl \
  --train-dir /path/to/giantmidi/train --val-dir /path/to/giantmidi/val \
  --output-dir runs/pretrain

uv run python -m mvi_v3.cli.train_baseline \
  --train-dir /path/to/maestro/train --val-dir /path/to/maestro/validation \
  --pretrained-backbone runs/pretrain/backbone.pt \
  --enable-controls --control-dims 2 --output-dir runs/ssl_finetune

# Build control regressor (for automatic inference)
uv run python -m mvi_v3.cli.build_presets \
  --train-dir /path/to/maestro/train --val-dir /path/to/maestro/validation \
  --stats runs/ssl_finetune/stats.json \
  --output-dir runs/ssl_finetune/presets_regress --method regress

# Evaluation
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir /path/to/maestro/test \
  --checkpoint runs/ssl_finetune/best.pt --stats runs/ssl_finetune/stats.json \
  --control-mode regression --preset-dir runs/ssl_finetune/presets_regress \
  --output-dir runs/ssl_finetune/eval_regression
```

## Control Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `oracle` | Ground truth controls from training stats | Upper bound evaluation |
| `regression` | RF regressor predicts controls from window features | **Production default** |
| `soft_preset` | Top-2 weighted blend of classifier probabilities | Smooth discrete alternative |
| `preset` | Hard classifier → nearest centroid | Discrete baseline |
| `default` | Fixed [0.5, 0.5] mid-range controls | Lower bound evaluation |

## Dependencies

- Python 3.12+, PyTorch, scikit-learn, joblib
- See `pyproject.toml` for full dependency list

## References

- Kim & Kim 2023: Seq2Seq + Luong Attention for velocity inference (v2)
- He et al. 2025: U-Net Colorizer approach, V-shaped loss, evaluation metrics
