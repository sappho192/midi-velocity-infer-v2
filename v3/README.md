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

### Ablation Study

| Configuration | MAE(reg) | CC | SD_ratio | MAE(oracle) |
|---|---|---|---|---|
| **Full system** | **9.81** | **0.717** | **85.3%** | **7.94** |
| − SSL pretraining | 10.24 | 0.689 | 83.3% | 8.30 |
| − Control conditioning | 9.60 | 0.734 | 91.7% | N/A |
| − V-shaped loss (β=0) | 9.87 | 0.709 | 81.0% | 7.91 |
| − All regularization | 9.79 | 0.716 | 84.7% | 7.95 |

SSL pretraining is the dominant contributor. See [docs/ablation-result.md](docs/ablation-result.md) for full results.

### Head Type Comparison

| Head Type | MAE(reg) | MAE(oracle) | SD_ratio |
|---|---|---|---|
| Regression (β=3) | 9.81 | 7.94 | 85.3% |
| Classification (expectation) | **9.67** | **7.63** | 76.2% |
| Stochastic (NLL) | 9.76 | 7.80 | 79.2% |

Classification head achieves the best MAE but lower SD_ratio.

## Architecture

- **Backbone**: 4-layer Transformer encoder (d=128, 4 heads)
- **Input features**: Pitch embedding + register bucket + 6 continuous features (onset, duration, IOI, chord_size, local_density, local_pitch_range)
- **Control conditioning**: 2D control params [expressiveness, dynamics_center] broadcast-added to token embeddings
- **Output heads**: Regression (Huber + V-shaped β=3), Classification (CE + expectation decode), Stochastic (Gaussian NLL)
- **SSL pretraining**: Masked Note Modeling on GiantMIDI-Piano (5,959 pieces) → MAESTRO fine-tune
- **Control predictor**: Lightweight MLP (21→64→64→2, 26KB) replaces RandomForest regressor (890MB)

## Project Structure

```
v3/
├── mvi_v3/
│   ├── cli/
│   │   ├── train_baseline.py      # Supervised training CLI (wandb integration)
│   │   ├── eval_baseline.py       # Evaluation (oracle/default/regression/regression_mlp/preset/soft_preset)
│   │   ├── build_presets.py       # Control preset builder (classify/regress/sweep)
│   │   ├── train_control_mlp.py   # Control predictor MLP training + checkpoint merge
│   │   └── pretrain_ssl.py        # SSL pretraining CLI
│   ├── data/
│   │   ├── ingest.py              # MIDI CSV loading
│   │   ├── events.py              # NoteEvent, Window, DatasetStats
│   │   ├── features.py            # Derived feature computation
│   │   ├── windowing.py           # Sliding window construction
│   │   ├── datasets.py            # PyTorch Dataset
│   │   ├── preset_features.py     # 21-dim window feature extraction
│   │   ├── pretrain_dataset.py    # MNM masking dataset
│   │   ├── normalize.py           # Velocity normalization
│   │   └── augmentation.py        # Data augmentation
│   ├── models/
│   │   ├── transformer.py         # TransformerVelocityModel
│   │   ├── embedding.py           # Pitch + register embeddings
│   │   ├── position.py            # T5 relative position bias
│   │   ├── conditioning.py        # Control conditioning layer
│   │   ├── heads.py               # Regression/Classification/Stochastic heads
│   │   ├── control_predictor.py   # Lightweight MLP control predictor (replaces RF)
│   │   └── pretrain_model.py      # SSL PretrainModel
│   ├── training/
│   │   ├── engine.py              # Training loop + validation metrics
│   │   ├── pretrain_engine.py     # SSL training loop
│   │   ├── losses.py              # Huber + V-shaped weighting + CE + Gaussian NLL
│   │   ├── ema.py                 # Exponential Moving Average
│   │   ├── checkpointing.py       # Checkpoint save/load
│   │   └── monitoring.py          # Training logging
│   ├── eval/
│   │   ├── metrics.py             # He2025 evaluation metrics
│   │   └── reconstruct.py         # Window → piece reconstruction
│   ├── io/
│   │   └── artifacts.py           # JSON I/O
│   └── config.py                  # BaselineConfig dataclass
├── docs/
│   ├── ablation-result.md         # Ablation study results (9 configurations)
│   ├── control-head-impl.md       # Control predictor MLP implementation
│   ├── control-head-plan.md       # Control head internalization analysis
│   ├── impl-status.md             # Full experiment log
│   └── ...
└── pyproject.toml
```

## Usage

### Training

```bash
# 1. SSL pretraining
uv run python -m mvi_v3.cli.pretrain_ssl \
  --train-dir /path/to/giantmidi/train --val-dir /path/to/giantmidi/val \
  --output-dir runs/pretrain

# 2. Supervised fine-tuning (with wandb logging)
uv run python -m mvi_v3.cli.train_baseline \
  --train-dir /path/to/maestro/train --val-dir /path/to/maestro/validation \
  --pretrained-backbone runs/pretrain/backbone.pt \
  --enable-controls --control-dims 2 \
  --dropout 0.2 --embedding-dropout 0.1 \
  --augment-velocity-jitter 2.0 --velocity-weight-beta 3.0 \
  --wandb-project mvi-v3 --wandb-run-name ssl_finetune \
  --output-dir runs/ssl_finetune

# 3. Train control predictor MLP (replaces 890MB RF with 26KB MLP)
uv run python -m mvi_v3.cli.train_control_mlp train \
  --train-dir /path/to/maestro/train --val-dir /path/to/maestro/validation \
  --stats runs/ssl_finetune/stats.json \
  --output-dir runs/ssl_finetune/control_mlp
```

### Evaluation

```bash
# Automatic control prediction (MLP, recommended)
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir /path/to/maestro/test \
  --checkpoint runs/ssl_finetune/best.pt --stats runs/ssl_finetune/stats.json \
  --control-mode regression_mlp --preset-dir runs/ssl_finetune/control_mlp \
  --output-dir runs/ssl_finetune/eval_regression_mlp

# Oracle controls (upper bound)
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir /path/to/maestro/test \
  --checkpoint runs/ssl_finetune/best.pt --stats runs/ssl_finetune/stats.json \
  --control-mode oracle \
  --output-dir runs/ssl_finetune/eval_oracle
```

### Single-file Deployment

```bash
# Merge MLP into model checkpoint
uv run python -m mvi_v3.cli.train_control_mlp merge \
  --checkpoint runs/ssl_finetune/best.pt \
  --mlp-checkpoint runs/ssl_finetune/control_mlp/control_mlp.pt

# Eval with merged checkpoint (no --preset-dir needed)
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir /path/to/maestro/test \
  --checkpoint runs/ssl_finetune/best.pt --stats runs/ssl_finetune/stats.json \
  --control-mode regression_mlp \
  --output-dir runs/ssl_finetune/eval_regression_mlp
```

## Control Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `oracle` | Ground truth controls from test set | Upper bound evaluation |
| `regression_mlp` | MLP predicts controls from window features (26KB) | **Production default** |
| `regression` | RF regressor predicts controls (890MB, legacy) | Backward compatibility |
| `soft_preset` | Top-2 weighted blend of classifier probabilities | Smooth discrete alternative |
| `preset` | Hard classifier → nearest centroid | Discrete baseline |
| `default` | Fixed [0.5, 0.5] mid-range controls | Lower bound evaluation |

## Dependencies

- Python 3.12+, PyTorch, wandb
- Optional: scikit-learn, joblib (only for legacy `regression`/`preset` modes)
- See `pyproject.toml` for full dependency list

## References

- Taein Kim & Yunho Kim 2023: Piano Velocity Prediction Using a Seq2Seq Model with Attention Mechanism
- Zhanhong He et al. 2025: Filling MIDI Velocity using U-Net Image Colorizer
  - U-Net Colorizer approach, V-shaped loss, evaluation metrics

# License

The code is released under the MIT License.
The checkpoints are released for academic or non-commercial use only. [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)