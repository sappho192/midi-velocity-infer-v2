# HDBSCAN 기반 Control Preset 자동 선택 파이프라인

## Context

SSL fine-tuned 모델의 Oracle MAE 7.94 vs Default [0.5,0.5] MAE 14.60 — gap이 크다. Default 단일 값 대신, train split의 oracle controls를 HDBSCAN 클러스터링하여 K개 프리셋을 도출하고, 비velocity 특성 기반 classifier로 inference 시 자동 선택. Test set 정보 누수 없이 train 데이터만으로 프리셋 구축.

**목표**: Default MAE 14.60 → Oracle MAE 7.94 사이의 gap을 줄이면서, fully automatic inference 유지.

---

## 파이프라인 개요

```
[Train windows] → HDBSCAN(oracle_controls 2D) → K presets (centroids)
                → extract_window_features(21-dim) → RandomForest(features → cluster_label)
                → save presets.json + classifier.joblib

[Test window]  → extract_window_features(21-dim) → classifier.predict() → preset centroid
               → model.forward(control_params=preset) → velocity prediction
```

---

## 구현 단계

### Step 1: `mvi_v3/data/preset_features.py` (NEW)

Window-level 21차원 aggregate feature 추출 함수:

```python
def extract_window_features(window: WindowRecord, window_size: int = 256) -> np.ndarray:
```

| 그룹 | 차원 | 설명 |
|------|------|------|
| Pitch stats | 4 | mean, std, min, max (÷127 정규화) |
| Continuous stats | 12 | 6 features × (mean, std) across valid notes |
| Fill ratio | 1 | true_length / window_size |
| Register dist | 4 | 4 buckets의 비율 |
| **Total** | **21** | |

continuous는 이미 z-normalized 상태이므로 직접 사용.

### Step 2: `mvi_v3/cli/build_presets.py` (NEW)

End-to-end 클러스터링 + classifier 훈련 스크립트.

**CLI args**: `--train-dir, --stats, --output-dir, --val-dir(optional), --min-cluster-size(50), --min-samples(25)`

**파이프라인**:
1. Train pieces 로딩 → windows 생성 (enable_controls=True) → oracle controls 정규화
2. `sklearn.cluster.HDBSCAN(min_cluster_size=50)` on `[N, 2]` normalized oracle controls
3. Noise (label=-1) → nearest centroid로 재할당
4. K개 preset centroids = cluster별 oracle_controls mean
5. 각 window에서 `extract_window_features()` → `[N, 21]` feature matrix
6. `RandomForestClassifier(n_estimators=200, class_weight='balanced')` 훈련
7. (Optional) val windows로 classifier accuracy + nearest-preset 기준 정확도 보고

**출력**:
- `presets.json`: centroids, cluster sizes, HDBSCAN params
- `classifier.joblib`: sklearn 모델
- `preset_report.json`: 클러스터링 통계, classifier accuracy

**Fallback**: HDBSCAN이 K≤1 반환 시 KMeans(K=5)로 대체 + 경고 출력.

### Step 3: `mvi_v3/cli/eval_baseline.py` (MODIFY)

**추가 args**:
- `--control-mode`: `oracle` (기존) / `default` / `preset`
- `--preset-dir`: presets.json + classifier.joblib 디렉토리

**Control mode 분기** (line 84-87 영역 수정):
```python
if control_mode == "oracle":
    normalize_oracle_controls(windows, stats.oracle_mins, stats.oracle_maxs)
elif control_mode == "preset":
    # classifier로 각 window의 preset 예측 → oracle_controls 덮어쓰기
    features = np.stack([extract_window_features(w) for w in windows])
    labels = classifier.predict(features)
    for w, label in zip(windows, labels):
        w.oracle_controls = np.array(presets[label], dtype=np.float32)
elif control_mode == "default":
    # oracle_controls를 None으로 설정 → ControlEmbedding이 학습된 default 사용
    for w in windows:
        w.oracle_controls = None
```

**중요**: preset centroids는 이미 [0,1] normalized 상태 → 추가 normalize 불필요.

기존 `--control-mode` 미지정 시: `enable_controls`이면 `oracle`, 아니면 해당 없음 (backward compatible).

---

## 의존성

`pyproject.toml`에 추가:
```toml
preset = [
  "scikit-learn>=1.3",
  "joblib>=1.3",
]
```

`scikit-learn >= 1.3`이 HDBSCAN, RandomForest 모두 포함. `joblib`은 sklearn 의존성이지만 명시 추가.

---

## 실행 순서

```bash
# 1. 프리셋 구축 (train data only)
uv run python -m mvi_v3.cli.build_presets \
  --train-dir /home/tikim/dataset/maestro/maestro-raw/maestro-midi/train \
  --val-dir /home/tikim/dataset/maestro/maestro-raw/maestro-midi/validation \
  --stats runs/ssl_finetune/stats.json \
  --output-dir runs/ssl_finetune/presets

# 2. 3-way 비교 평가
TEST_DIR=/home/tikim/dataset/maestro/maestro-raw/maestro-midi/test
CKPT=runs/ssl_finetune/best.pt
STATS=runs/ssl_finetune/stats.json

# Oracle (upper bound)
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST_DIR --checkpoint $CKPT --stats $STATS \
  --control-mode oracle --output-dir runs/ssl_finetune/eval_oracle

# Default (current baseline)
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST_DIR --checkpoint $CKPT --stats $STATS \
  --control-mode default --output-dir runs/ssl_finetune/eval_default

# Preset (자동 선택)
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST_DIR --checkpoint $CKPT --stats $STATS \
  --control-mode preset --preset-dir runs/ssl_finetune/presets \
  --output-dir runs/ssl_finetune/eval_preset
```

---

## 검증

1. **클러스터링 품질**: K가 3~15 범위인지, noise 비율 < 20%인지 확인
2. **Classifier accuracy**: val set nearest-preset 기준 accuracy > 50% (random = 1/K)
3. **최종 결과**: Preset MAE가 Default(14.60)보다 유의미하게 낮은지, gap closure ratio = (14.60 - preset_MAE) / (14.60 - 7.94)

---

## 파일 변경 요약

| 파일 | 액션 | 설명 |
|------|------|------|
| `mvi_v3/data/preset_features.py` | NEW | 21-dim window feature 추출 |
| `mvi_v3/cli/build_presets.py` | NEW | HDBSCAN 클러스터링 + RF classifier 훈련 |
| `mvi_v3/cli/eval_baseline.py` | MODIFY | `--control-mode`, `--preset-dir` 추가 |
| `pyproject.toml` | MODIFY | `preset` optional dependency 추가 |
