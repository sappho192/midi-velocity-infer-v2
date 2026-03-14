# v3 Implementation Status

Last updated: 2026-03-14

## Completed

### Training Infrastructure
- [x] AdamW optimizer + cosine schedule with linear warmup
- [x] Checkpoint/resume (`best.pt`, `latest.pt`) with atomic saves
- [x] RNG state preservation for exact reproducibility
- [x] Training monitoring (`status.json`, `training.jsonl`, console output)
- [x] Early stopping (patience-based)
- [x] EMA (Exponential Moving Average)
  - Warmup: `decay = min(0.999, 1 - 1/step)`
  - EMA state saved/restored in checkpoints
  - `--no-ema` flag for eval with training weights
- [x] Data pipeline parallelization (`ProcessPoolExecutor` for CSV loading + feature computation)
- [x] Batch size default: 256 (VRAM 사용량 ~2.7/16GB로 여유 확인)

### Loss Function
- [x] V-shaped loss weighting (He et al. 2025 inspired)
  - `w = 1 + β|v - 0.5|` — 극단 velocity에 더 큰 가중치
  - 기본값 `β=3.0`, CLI `--velocity-weight-beta`로 조절 가능
- [x] Classification cross-entropy loss
  - `masked_cross_entropy_loss` with label smoothing + V-shaped weighting 지원

### Output Heads
- [x] **Regression** (`--head-type regression`): Linear → scalar, Huber loss
- [x] **Classification** (`--head-type classification`): Linear → 128-bin logits, CE loss
  - Decoding: `--decode-mode expectation` (soft) / `argmax` (sharp)
  - `--label-smoothing` (기본 0.1)
- [x] **Stochastic** (`--head-type stochastic`): (mu, log_sigma) 출력, Gaussian NLL

### Evaluation Metrics (He et al. 2025 체계)

`compute_piece_metrics()` → `aggregate_metrics()`로 per-piece + 집계 산출.

| Metric | 설명 | Threshold |
|--------|------|-----------|
| MAE | Mean Absolute Error (raw 0-127) | — |
| MSE | Mean Squared Error | — |
| SD_velo | `std(pred)` — 예측 다이나믹 레인지 | — |
| SD_ratio | `pred_std / true_std` — 1.0이 이상적 | — |
| SD_ae | `std(|pred - true|)` — 에러 일관성 | — |
| CC | Pearson correlation coefficient | — |
| Recall(10%) | `|err| < 12.7` 비율 | 12.7 (=127×10%) |
| Recall(5%) | `|err| < 6.4` 비율 | 6.4 (=127×5%) |

집계: macro (piece 평균) + weighted (note 수 가중 평균) 두 방식 모두 출력.

### Training Results (MAESTRO medium, 200 train / 32 val pieces)

모든 수치는 macro 평균 (piece 단위). No-EMA = training weights 기준.

#### Regression Head

| Config | MAE | SD_velo | SD_ratio | SD_ae | CC | Recall(10%) | Recall(5%) |
|--------|-----|---------|----------|-------|----|-------------|------------|
| β=0 (EMA) | 11.82 | 10.45 | 58.1% | 9.93 | 0.577 | 62.0% | 34.5% |
| β=0 (No EMA) | 11.50 | 11.46 | 63.6% | 9.47 | 0.577 | 62.8% | 34.9% |
| β=3 (EMA) | 11.75 | 11.52 | 64.3% | 9.91 | 0.561 | 61.8% | 34.3% |
| β=3 (No EMA) | 11.55 | 12.33 | 68.4% | 9.62 | 0.572 | 62.4% | 34.6% |

#### Classification Head (128-bin, β=3, label_smoothing=0.1)

| Config | MAE | SD_velo | SD_ratio | SD_ae | CC | Recall(10%) | Recall(5%) |
|--------|-----|---------|----------|-------|----|-------------|------------|
| expectation (EMA) | 12.18 | 8.30 | 46.3% | 9.62 | 0.537 | 58.6% | 31.4% |
| expectation (No EMA) | 11.45 | 11.18 | 62.0% | 9.57 | 0.571 | 63.0% | 34.8% |
| **argmax (No EMA)** | 12.41 | **16.28** | **90.6%** | 11.45 | 0.530 | 59.9% | 35.1% |

참고: He2025 MAESTRO test 기준 MAE=11.5, SD_velo=10.7.

### Full MAESTRO Results (962 train / 137 val pieces)

모든 수치는 macro 평균 (piece 단위). β=3, No-EMA 기준.

#### Regression Head (β=3)

| Config | MAE | SD_velo | SD_ratio | SD_ae | CC | Recall(10%) | Recall(5%) |
|--------|-----|---------|----------|-------|----|-------------|------------|
| EMA | 10.15 | 15.34 | 85.4% | 8.75 | 0.690 | 70.0% | 41.3% |
| No EMA | 10.09 | 15.21 | 84.7% | 8.71 | 0.691 | 70.2% | 41.5% |

#### Classification Head (128-bin, β=3, label_smoothing=0.1)

| Config | MAE | SD_velo | SD_ratio | SD_ae | CC | Recall(10%) | Recall(5%) |
|--------|-----|---------|----------|-------|----|-------------|------------|
| expectation (No EMA) | **9.91** | 13.42 | 74.7% | **8.49** | **0.693** | **71.3%** | 41.9% |
| argmax (No EMA) | 10.63 | **17.02** | **94.9%** | 9.92 | 0.662 | 68.7% | **42.4%** |

참고: He2025 MAESTRO test 기준 MAE=11.5, SD_velo=10.7.

### 발견 및 분석

**데이터 규모 효과 (medium → full)**
- MAE: 11.5 → **9.91** (14% 개선, He2025 대비 우위 확정)
- SD_ratio: 68% → **94.9%** (argmax, 거의 완벽한 다이나믹 레인지)
- CC: 0.57 → **0.693** (상관관계 대폭 향상)
- 모든 메트릭에서 일관된 개선 — 데이터 확장이 가장 큰 단일 개선 요인

**V-shaped loss weighting (β=3)**
- regression head에서 SD_ratio 58.1% → 68.4%로 개선 (medium), MAE 열화 미미
- 극단 velocity 예측 능력 향상 확인

**Classification head**
- **argmax**: SD_ratio **94.9%**로 다이나믹 레인지 거의 완전 회복. MAE는 10.63으로 regression 대비 소폭 상승
- **expectation**: MAE **9.91** (전체 최저), CC **0.693** (전체 최고). SD_ratio 74.7%로 regression(84.7%)보다 낮음
- medium에서는 epoch 15에서 조기 종료했으나 full에서는 epoch 36까지 학습 — 데이터가 많으면 classification도 더 잘 수렴

**EMA 관련**
- Full MAESTRO regression에서도 No-EMA가 EMA보다 소폭 우수
- best checkpoint 선정이 training weights 기준 val_loss로 되어 있어 EMA weights의 최적 시점과 불일치
- 향후: EMA weights로 validation 평가하는 방식 검토 필요

**Head 선택 가이드**
- 정확도 우선: classification + expectation (MAE 9.91, CC 0.693)
- 표현력 우선: classification + argmax (SD_ratio 94.9%, Recall(5%) 42.4%)
- 균형: regression β=3 (MAE 10.09, SD_ratio 84.7%, 모든 메트릭에서 안정적)

### Test Set Results (177 pieces)

MAESTRO test split, weighted 평균 (note 수 가중). β=3 for all v3 models.

#### v3 Regression Head (β=3)

| Config | MAE | MSE | SD_velo | SD_ratio | SD_ae | CC | Recall(10%) | Recall(5%) |
|--------|-----|-----|---------|----------|-------|----|-------------|------------|
| No EMA | 26.58 | 969.77 | 6.10 | 35.0% | 15.41 | 0.1510 | 23.6% | 11.6% |
| EMA | 27.63 | 1033.57 | 5.84 | 33.5% | 15.59 | 0.1487 | 21.7% | 10.5% |

#### v3 Classification Head (128-bin, β=3, label_smoothing=0.1)

| Config | MAE | MSE | SD_velo | SD_ratio | SD_ae | CC | Recall(10%) | Recall(5%) |
|--------|-----|-----|---------|----------|-------|----|-------------|------------|
| expectation (No EMA) | 23.43 | 764.68 | 4.19 | 24.0% | 13.90 | 0.2493 | 28.4% | 14.4% |
| argmax (No EMA) | 34.30 | 1470.20 | 4.38 | 25.0% | 16.31 | 0.1795 | 12.2% | 5.5% |

#### v2 Baseline (Seq2Seq + Luong Attention)

| Config | MAE | MSE | SD_velo | SD_ratio | SD_ae | CC | Recall(10%) | Recall(5%) |
|--------|-----|-----|---------|----------|-------|----|-------------|------------|
| v2 (ONNX) | 13.87 | 302.32 | 6.34 | 36.1% | 10.25 | 0.3439 | 52.4% | 28.5% |

#### Test vs Validation Gap 분석

| Model | Val MAE | Test MAE | Gap | Val SD_ratio | Test SD_ratio |
|-------|---------|----------|-----|--------------|---------------|
| v3 Reg NoEMA | 10.09 | 26.58 | +163% | 84.7% | 35.0% |
| v3 Cls Exp | 9.91 | 23.43 | +136% | 74.7% | 24.0% |
| v2 | — | 13.87 | — | — | 36.1% |

**심각한 과적합 문제 확인**:
- v3 모델은 validation → test에서 MAE 2.4~2.6배 증가, SD_ratio 절반 이하로 하락
- v2가 v3보다 test set에서 모든 메트릭에서 우수 (MAE 13.87 vs 23.43, CC 0.34 vs 0.25, Recall10% 52.4% vs 28.4%)
- Validation set 기준 결과 (MAE 9.91)는 He2025 대비 우수했으나, test set 일반화에 실패
- 원인 가설: (1) 학습 데이터와 val/test의 분포 차이, (2) transformer 모델의 과적합 경향, (3) piece-level windowing에서의 정보 누수 가능성

## Next Steps

1. ~~**Full MAESTRO 학습**~~ ✅ — regression + classification 모두 완료
2. ~~**Classification head**~~ ✅ — 128-bin classification 구현 및 실험 완료
3. ~~**Eval metric 보강**~~ ✅ — He2025 메트릭 체계 도입 완료
4. ~~**Test set 평가 + v2 비교**~~ ✅ — 과적합 문제 확인
5. **P0: 과적합 해결** — test set 일반화 성능 확보가 최우선 과제
6. **P0 research** — Canonical Note-Event Format 검증, v2-compatible eval contract 정리

### 과적합 해결 방안 후보
- Dropout 강화 / weight decay 조정
- Data augmentation (tempo/velocity scaling, 곡 분할 등)
- Train/val/test split 분포 분석 — MAESTRO 공식 split의 특성 확인
- 모델 크기 축소 (layer 수, hidden dim)
- Regularization: mixup, label smoothing 강화
- Cross-validation으로 과적합 정도 재확인

## Reference Papers

- **Kim & Kim 2023** (v2): Seq2Seq + Luong Attention. MSE/MAE 좋지만 F1 낮음 (SD 부족). 확률적 모델/분류 기반 접근 제안.
- **He et al. 2025** (U-Net Colorizer): SD_velo를 핵심 지표로 강조. V-shaped loss weighting, BCE + CosSim loss. MAESTRO test MAE=11.5, SD_velo=10.7.
