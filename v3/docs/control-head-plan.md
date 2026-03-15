# Control Head Internalization Plan

## 문제

현재 regression control mode는 외부 RandomForest regressor를 사용:

```
[Window Features (21-dim)] → RandomForest (890MB joblib) → [Controls (2-dim)] → Model
```

- `control_regressor.joblib` ~890MB — 모델 자체(~수 MB)보다 훨씬 큼
- 배포 시 비실용적

## 현재 아키텍처

Control은 **embedding 단계에서 모든 token에 additive bias**로 주입:

```python
# conditioning.py: ControlEmbedding
control_params [B, 2] → MLP(2 → d_model → d_model) → [B, 1, d_model] (broadcast add)
```

Backbone이 처리하는 representation 자체가 control에 의존함.

## 후보 방안

### Option A: Lightweight MLP head (권장)

```
[Window Features (21-dim)] → MLP (수 KB) → [Controls (2-dim)] → Model
```

- 모델 내부에 작은 MLP (21-dim → hidden → 2-dim) 추가
- 학습 시 oracle controls를 target으로 같이 학습 (MSE loss)
- 추론 시 window features에서 바로 control 예측 → 외부 regressor 불필요
- 추가 파라미터: ~수 KB
- 기존 checkpoint 호환: head만 추가하면 됨
- 현재 RandomForest val MAE 0.067 (normalized 0-1 scale)이므로 MLP로도 충분히 대체 가능

### Option B1: Two-pass inference

```
Pass 1: backbone(input, controls=default) → hidden → ControlPredictor → predicted_controls
Pass 2: backbone(input, controls=predicted_controls) → velocity prediction
```

- 학습: Pass 1의 ControlPredictor를 oracle target으로 학습 (stop-gradient)
- 추론: 2번 forward pass
- 단점: inference 비용 2배, 수렴 보장 없음

### Option B2: Split architecture (control-free backbone + late conditioning)

```
backbone(input, no controls) → hidden_states
├── ControlPredictor: pool(hidden) → MLP → predicted_controls [B, 2]
└── ConditionedHead: hidden + control_embedding(predicted_controls) → velocity
```

- Backbone을 control 없이 동작하도록 재설계 (현재 control은 embedding 단계에서 주입되므로 구조 변경 필요)
- Control 정보는 head 직전에만 주입
- 학습: ControlPredictor는 oracle target + MSE loss
- 단점: 기존 checkpoint 전부 호환 불가, backbone의 control 의존 표현력 손실 가능

### Option B3: Auxiliary encoder (lightweight)

```
ControlEncoder(input_features) → predicted_controls [B, 2]   ← 별도 경량 encoder
backbone(input, controls=predicted_controls) → velocity
```

- 별도의 작은 네트워크가 같은 input에서 control 예측
- 학습: oracle controls를 target으로, main model과 joint training
- 추론: ControlEncoder → backbone 순차 실행
- 단점: 추가 네트워크가 필요하지만, 본질적으로 Option A와 유사

## 비교표

| | Option A (MLP) | Option B1 (Two-pass) | Option B2 (Split arch) | Option B3 (Aux encoder) |
|---|---|---|---|---|
| 추가 파라미터 | ~수 KB | 0 (+ pooling layer) | 구조 재설계 | ~수 KB~수 MB |
| Inference 비용 | 1 pass + tiny MLP | 2 passes | 1 pass | 1 pass + small encoder |
| 기존 checkpoint 호환 | 호환 (head만 추가) | 호환 | 불가 | 부분 호환 |
| Input | 21-dim hand-crafted features | Backbone hidden states | Backbone hidden states | Raw note features |
| 표현력 | RandomForest 수준 | Backbone 급 | Backbone 급 | 설계에 따라 다름 |

## 결론

- B2는 아키텍처 재설계 + 재학습 비용이 큼
- B1은 inference 비용 2배
- B3는 결국 Option A의 변형
- **Option A가 가장 실용적**: 890MB → 수 KB, 기존 checkpoint 호환, 성능 유지 가능

---

## 구현 계획 (Option A)

대상: 기존 학습 완료된 3개 모델 (regression, classification, stochastic head).
세 모델의 `oracle_mins`/`oracle_maxs`가 동일 → **MLP 1개로 공유 가능**.

### 변경 파일

| File | Action | Description |
|------|--------|-------------|
| `mvi_v3/models/control_predictor.py` | **NEW** | `ControlPredictorMLP` 모듈 |
| `mvi_v3/cli/train_control_mlp.py` | **NEW** | MLP 학습 CLI |
| `mvi_v3/cli/eval_baseline.py` | **MODIFY** | `regression_mlp` control mode 추가 |
| `pyproject.toml` | **MODIFY** | CLI entry point 추가 |

### Step 1: ControlPredictorMLP 모듈

**NEW** `v3/mvi_v3/models/control_predictor.py`

- 구조: `21 → 64 → ReLU → 64 → ReLU → 2 → Sigmoid`
- Sigmoid: oracle controls가 min-max normalized [0,1]이므로 자연 제한
- 파라미터: ~5,698개 (~23KB)

### Step 2: 학습 CLI

**NEW** `v3/mvi_v3/cli/train_control_mlp.py`

데이터 로딩은 `build_presets.py`의 `_load_train_data()`, `_load_val_data()` 재사용 (import).

- Input: 43,727 train windows의 21-dim features (`preset_features.py: extract_window_features`)
- Target: 2-dim normalized oracle controls
- MSE loss, Adam, lr=1e-3, ReduceLROnPlateau
- ~200 epochs (CPU에서 수 초)
- 저장: `control_mlp.pt` (state_dict + config + val_mae)

```bash
uv run python -m mvi_v3.cli.train_control_mlp \
  --train-dir $TRAIN --val-dir $VAL \
  --stats runs/ssl_finetune/stats.json \
  --output-dir runs/ssl_finetune/control_mlp \
  --epochs 200 --lr 1e-3 --hidden-dim 64
```

### Step 3: eval_baseline.py 수정

`--control-mode` choices에 `"regression_mlp"` 추가.

새 elif 분기 (기존 `regression` 블록 이후):
```python
elif enable_controls and control_mode == "regression_mlp":
    from mvi_v3.models.control_predictor import ControlPredictorMLP
    mlp_ckpt = torch.load(preset_dir / "control_mlp.pt", map_location="cpu", weights_only=False)
    mlp = ControlPredictorMLP(**mlp_ckpt["config"])
    mlp.load_state_dict(mlp_ckpt["state_dict"])
    mlp.eval()
    features = np.stack([extract_window_features(w) for w in windows])
    with torch.no_grad():
        predicted = mlp(torch.as_tensor(features, dtype=torch.float32)).numpy()
    for w, ctrl in zip(windows, predicted):
        w.oracle_controls = ctrl.astype(np.float32)
```

sklearn 의존성 없이 PyTorch만으로 동작.

### Step 4: pyproject.toml entry point

```toml
mvi-v3-train-control-mlp = "mvi_v3.cli.train_control_mlp:main"
```

### Step 5: Checkpoint merge 유틸리티

학습된 MLP를 모델 checkpoint에 병합하여 단일 파일 배포 지원.

`eval_baseline.py`에서 `--preset-dir` 없이도 checkpoint 내부의 MLP를 자동 사용:
```python
# checkpoint에 control_mlp가 있으면 자동 사용
if enable_controls and control_mode == "regression_mlp" and args.preset_dir is None:
    if "control_mlp" in checkpoint:
        # checkpoint 내부 MLP 사용
        ...
```

Merge 스크립트 (`train_control_mlp.py`에 `--merge-into` 옵션 추가):
```bash
# MLP를 모델 checkpoint에 merge
uv run python -m mvi_v3.cli.train_control_mlp \
  --merge-into runs/ssl_finetune/best.pt \
  --mlp-checkpoint runs/ssl_finetune/control_mlp/control_mlp.pt
```

### 실행 및 검증

```bash
# 1. MLP 학습 (1번만, 세 모델 공유)
uv run python -m mvi_v3.cli.train_control_mlp \
  --train-dir $TRAIN --val-dir $VAL \
  --stats runs/ssl_finetune/stats.json \
  --output-dir runs/ssl_finetune/control_mlp

# 2. 세 모델 eval (별도 파일 모드)
for RUN in runs/ssl_finetune runs_ablation/cls_head runs_ablation/stoch_head; do
  uv run python -m mvi_v3.cli.eval_baseline \
    --data-dir $TEST --checkpoint $RUN/best.pt --stats $RUN/stats.json \
    --control-mode regression_mlp \
    --preset-dir runs/ssl_finetune/control_mlp \
    --output-dir $RUN/eval_regression_mlp
done

# 3. RF 대비 parity 확인 (regression_mlp vs regression → MAE 차이 < 0.1)

# 4. (선택) Merge 후 단일 파일 eval
uv run python -m mvi_v3.cli.train_control_mlp \
  --merge-into runs/ssl_finetune/best.pt \
  --mlp-checkpoint runs/ssl_finetune/control_mlp/control_mlp.pt

uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST --checkpoint runs/ssl_finetune/best.pt --stats ... \
  --control-mode regression_mlp \
  --output-dir runs/ssl_finetune/eval_regression_mlp_merged
```

검증 기준:
1. MLP val control MAE ≤ RF val MAE (0.067)
2. End-to-end velocity MAE 차이 < 0.1 (regression_mlp vs regression)
3. `control_mlp.pt` 크기 < 100KB (vs RF 890MB)
4. 세 모델 모두 동일 MLP로 정상 동작
5. Merged checkpoint eval 결과가 별도 파일 모드와 동일
