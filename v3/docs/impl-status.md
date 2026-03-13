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
  - 결과: SD ratio 56.4% → 64.2% (평균 회귀 완화), MAE/RMSE 유지

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

### Baseline Training Results (MAESTRO medium, 200 train / 32 val pieces)

| Config | MAE | SD_velo | SD_ratio | SD_ae | CC | Recall(10%) | Recall(5%) |
|--------|-----|---------|----------|-------|----|-------------|------------|
| β=0 (EMA) | 11.82 | 10.45 | 58.1% | 9.93 | 0.577 | 62.0% | 34.5% |
| β=0 (No EMA) | 11.50 | 11.46 | 63.6% | 9.47 | 0.577 | 62.8% | 34.9% |
| β=3 (EMA) | 11.75 | 11.52 | 64.3% | 9.91 | 0.561 | 61.8% | 34.3% |
| β=3 (No EMA) | 11.55 | **12.33** | **68.4%** | 9.62 | 0.572 | 62.4% | 34.6% |

참고: He2025 MAESTRO test 기준 MAE=11.5, SD_velo=10.7.

EMA가 best checkpoint 선정 시 training weights 기준 val_loss로 판단하므로,
EMA weights의 최적 시점과 일치하지 않을 수 있음. 향후 EMA weights로 validation 평가하는 방식 검토 필요.

## Next Steps (우선순위 선택지)

1. **Full MAESTRO (962 pieces) 학습** — medium이 아닌 전체 데이터로 baseline 성능 확정
2. **Classification head** — 128-bin classification으로 평균 회귀 추가 완화. config에 `stochastic_head` 옵션 이미 존재
3. ~~**Eval metric 보강**~~ ✅ — He2025 메트릭 체계 도입 완료
4. **P0 research** — Canonical Note-Event Format 검증, v2-compatible eval contract 정리

### 향후 최적화
- Optuna를 이용한 하이퍼파라미터 탐색 (velocity_weight_beta, learning_rate, batch_size 등)
  - 모델 아키텍처와 loss 설계가 안정화된 후에 실행 예정

## Reference Papers

- **Kim & Kim 2023** (v2): Seq2Seq + Luong Attention. MSE/MAE 좋지만 F1 낮음 (SD 부족). 확률적 모델/분류 기반 접근 제안.
- **He et al. 2025** (U-Net Colorizer): SD_velo를 핵심 지표로 강조. V-shaped loss weighting, BCE + CosSim loss. MAESTRO test MAE=11.5, SD_velo=10.7.
