# Control Predictor MLP — 구현 결과

## 요약

외부 RandomForest regressor (`control_regressor.joblib`, ~890MB)를 경량 PyTorch MLP (`control_mlp.pt`, 26KB)로 대체. 파일 크기 34,000배 축소, sklearn 의존성 제거, 성능 동등.

## Control Prediction 정확도

| | RandomForest | MLP |
|---|---|---|
| Val control MAE | 0.0671 | **0.0650** |
| 파일 크기 | ~890MB | **26KB** |
| sklearn 필요 | Yes | **No** |

MLP 구조: `21 → 64 → ReLU → 64 → ReLU → 2 → Sigmoid` (5,698 파라미터)

## End-to-end Velocity MAE Parity

`regression_mlp` (MLP) vs `regression` (RF) 비교. MAESTRO test set (177 pieces, 741,410 notes).

| Model | MAE (RF) | MAE (MLP) | Δ | CC (RF) | CC (MLP) | SD_ratio |
|---|---|---|---|---|---|---|
| Regression (β=3) | 9.81 | 9.87 | +0.06 | 0.717 | 0.719 | 85.3% |
| Classification (exp) | 9.67 | 9.70 | +0.03 | 0.722 | 0.723 | 76.3% |
| Stochastic (NLL) | 9.76 | 9.82 | +0.06 | 0.715 | 0.716 | 79.3% |

모든 모델에서 MAE 차이 < 0.1. 세 모델 동일 MLP 공유 (oracle stats 동일).

## 생성/수정 파일

| File | Action | Description |
|---|---|---|
| `mvi_v3/models/control_predictor.py` | **NEW** | `ControlPredictorMLP` 모듈 |
| `mvi_v3/cli/train_control_mlp.py` | **NEW** | 학습 CLI (train/merge 서브커맨드) |
| `mvi_v3/cli/eval_baseline.py` | **MODIFY** | `regression_mlp` control mode 추가 |
| `pyproject.toml` | **MODIFY** | `mvi-v3-train-control-mlp` entry point |

## 사용법

### MLP 학습

```bash
uv run python -m mvi_v3.cli.train_control_mlp train \
  --train-dir $TRAIN --val-dir $VAL \
  --stats runs/ssl_finetune/stats.json \
  --output-dir runs/ssl_finetune/control_mlp \
  --epochs 200 --lr 1e-3 --hidden-dim 64
```

### Eval (별도 파일 모드)

```bash
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST --checkpoint $RUN/best.pt --stats $RUN/stats.json \
  --control-mode regression_mlp \
  --preset-dir runs/ssl_finetune/control_mlp \
  --output-dir $RUN/eval_regression_mlp
```

### Checkpoint merge (단일 파일 배포)

```bash
# MLP를 모델 checkpoint에 병합
uv run python -m mvi_v3.cli.train_control_mlp merge \
  --checkpoint runs/ssl_finetune/best.pt \
  --mlp-checkpoint runs/ssl_finetune/control_mlp/control_mlp.pt

# Merged checkpoint eval (--preset-dir 불필요, 자동으로 내부 MLP 사용)
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST --checkpoint runs/ssl_finetune/best.pt --stats ... \
  --control-mode regression_mlp \
  --output-dir runs/ssl_finetune/eval_regression_mlp
```

## Output artifact

```
runs/ssl_finetune/control_mlp/
├── control_mlp.pt           (26,053 bytes)
└── control_mlp_report.json
```
