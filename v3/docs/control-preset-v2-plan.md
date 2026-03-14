# Preset 시스템 개선: Regression + Sweep K + Soft Ensemble

## Context

현재 HDBSCAN K=3 preset 시스템 결과:
- **Oracle MAE 7.94 → Preset MAE 11.68 → Default MAE 14.60**
- Gap closure 43.8% (weighted), 150/177 pieces에서 default보다 개선
- 그러나 **27/177 pieces에서 preset이 default보다 나쁨** (잘못된 preset 배정)
- 클러스터 극심한 불균형: Cluster 1이 83% (36,300/43,727 windows)
- Val classifier accuracy 65.2% (random 33.3%), noise 28.8%

**핵심 병목 3가지**:
1. **Discretization loss**: 연속 2D 공간을 3개 centroid로 양자화 → 정보 손실
2. **Classifier 정확도 한계**: 21-dim features → velocity-based controls 예측이 본질적으로 어려움 (velocity 자체를 보지 못함)
3. **클러스터 불균형**: K=3에서 83%가 하나의 클러스터 → 세밀한 구분 불가

**목표**: Preset MAE 11.68을 추가 개선하되, fully automatic inference 유지. 복수의 독립적 개선을 구현하여 ablation 비교.

---

## 개선 전략 (우선순위순)

### A. Direct Regression (discretization loss 제거) — 최우선

Preset classification 대신 21-dim features → [expressiveness, dynamics_center] 직접 회귀.
- RandomForestRegressor로 연속값 예측 → 양자화 손실 제거
- 구현 난이도 낮음 (classifier → regressor 교체)

### B. KMeans K sweep (K=3,5,7,10) — A와 독립 실행

HDBSCAN K=3의 극심한 불균형 해소. KMeans로 K를 5~10까지 실험.
- 각 K에서 classifier accuracy + nearest-preset MAE 보고
- 최적 K 선정 후 preset 평가

### C. Soft Ensemble (Top-2 weighted blending) — A/B 위에 적용 가능

Hard assignment 대신 classifier의 predict_proba로 top-2 preset weighted average.
- Preset 경계의 window들에서 smooth interpolation
- 구현 매우 간단 (eval_baseline.py만 수정)

---

## 구현 단계

### Step 1: `build_presets.py`에 regression 모드 추가

`--method` arg: `classify` (현재) / `regress` (새로 추가)

```
regress 모드:
  1. Train windows 로딩 + oracle controls 정규화 (기존과 동일)
  2. Feature matrix [N, 21] 추출 (기존과 동일)
  3. RandomForestRegressor(n_estimators=200) 훈련: X=[N,21] → y=[N,2]
  4. Val set에서 predicted controls vs oracle controls MAE 보고
  5. control_regressor.joblib 저장
```

**출력**: `control_regressor.joblib`, `regression_report.json`

### Step 2: `build_presets.py`에 KMeans K sweep 추가

`--method sweep` 모드:

```
sweep 모드:
  1. K ∈ {3, 5, 7, 10}에 대해 KMeans 실행
  2. 각 K에서:
     - RF classifier 훈련 + val accuracy 보고
     - Nearest-preset oracle MAE (ceiling) 계산
     - Cluster size 분포 + max_cluster_ratio 보고
  3. 최적 K 자동 선정 (val accuracy × nearest-preset MAE 기준)
  4. 최적 K의 presets.json + classifier.joblib 저장
```

### Step 3: `eval_baseline.py`에 regression + soft_preset 모드 추가

`--control-mode` choices 확장: `oracle / default / preset / regression / soft_preset`

```python
# regression 모드
elif control_mode == "regression":
    regressor = joblib.load(preset_dir / "control_regressor.joblib")
    features = np.stack([extract_window_features(w) for w in windows])
    predicted = np.clip(regressor.predict(features), 0.0, 1.0)
    for w, ctrl in zip(windows, predicted):
        w.oracle_controls = ctrl.astype(np.float32)

# soft_preset 모드
elif control_mode == "soft_preset":
    clf = joblib.load(preset_dir / "classifier.joblib")
    features = np.stack([extract_window_features(w) for w in windows])
    proba = clf.predict_proba(features)  # [N, K]
    centroid_array = np.array(presets)
    for w, p in zip(windows, proba):
        top2 = np.argsort(p)[-2:][::-1]
        weights = p[top2] / p[top2].sum()
        blended = weights[0] * centroid_array[top2[0]] + weights[1] * centroid_array[top2[1]]
        w.oracle_controls = np.clip(blended, 0.0, 1.0).astype(np.float32)
```

---

## 파일 변경 요약

| 파일 | 액션 | 설명 |
|------|------|------|
| `mvi_v3/cli/build_presets.py` | MODIFY | `--method` arg 추가 (classify/regress/sweep) |
| `mvi_v3/cli/eval_baseline.py` | MODIFY | regression, soft_preset control modes 추가 |

기존 파일 2개만 수정. 새 파일 없음.

---

## 실행 및 검증

```bash
TRAIN=/home/tikim/dataset/maestro/maestro-raw/maestro-midi/train
VAL=/home/tikim/dataset/maestro/maestro-raw/maestro-midi/validation
TEST=/home/tikim/dataset/maestro/maestro-raw/maestro-midi/test
STATS=runs/ssl_finetune/stats.json
CKPT=runs/ssl_finetune/best.pt

# A. Regression
uv run python -m mvi_v3.cli.build_presets \
  --train-dir $TRAIN --val-dir $VAL --stats $STATS \
  --output-dir runs/ssl_finetune/presets_regress --method regress

uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST --checkpoint $CKPT --stats $STATS \
  --control-mode regression --preset-dir runs/ssl_finetune/presets_regress \
  --output-dir runs/ssl_finetune/eval_regression

# B. K sweep (최적 K 자동 선정)
uv run python -m mvi_v3.cli.build_presets \
  --train-dir $TRAIN --val-dir $VAL --stats $STATS \
  --output-dir runs/ssl_finetune/presets_sweep --method sweep

uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST --checkpoint $CKPT --stats $STATS \
  --control-mode preset --preset-dir runs/ssl_finetune/presets_sweep \
  --output-dir runs/ssl_finetune/eval_sweep

# C. Soft ensemble (K=3 기존 presets 위에)
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST --checkpoint $CKPT --stats $STATS \
  --control-mode soft_preset --preset-dir runs/ssl_finetune/presets \
  --output-dir runs/ssl_finetune/eval_soft

# D. Soft ensemble on sweep presets
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST --checkpoint $CKPT --stats $STATS \
  --control-mode soft_preset --preset-dir runs/ssl_finetune/presets_sweep \
  --output-dir runs/ssl_finetune/eval_soft_sweep
```

### 기대 결과 비교표

| Mode | Expected MAE | 근거 |
|------|-------------|------|
| Oracle | 7.94 | Upper bound |
| Regression | ~10–11 | Discretization 제거, but feature ceiling 존재 |
| Soft preset (sweep) | ~10.5–11.5 | K 증가 + interpolation |
| Preset (sweep K) | ~10.5–11.5 | 더 세밀한 클러스터 |
| Soft preset (K=3) | ~11.0–11.5 | 기존 K=3에 interpolation만 추가 |
| Preset (K=3) | 11.68 | 현재 baseline |
| Default [0.5,0.5] | 14.60 | Lower bound |

### 성공 기준
- Regression 또는 sweep 중 하나 이상이 Preset K=3 (11.68)보다 유의미하게 개선
- Gap closure > 50% (MAE < 11.27) 달성
