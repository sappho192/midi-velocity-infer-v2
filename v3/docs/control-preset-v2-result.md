# Control Preset v2: Regression + Sweep K + Soft Ensemble 결과

## 실험 배경

기존 HDBSCAN K=3 preset 시스템의 한계:
- Oracle MAE 7.94 → Preset MAE 11.68 → Default MAE 14.60
- Gap closure 43.8%, 27/177 pieces에서 preset이 default보다 나쁨
- 클러스터 극심한 불균형 (Cluster 1이 83%)

**핵심 병목**: Discretization loss (연속 2D → 3개 centroid 양자화), Classifier 정확도 한계, 클러스터 불균형

## 개선 방법

### A. Direct Regression
- RandomForestRegressor(n_estimators=200): 21-dim features → [expressiveness, dynamics_center] 직접 회귀
- Discretization loss 완전 제거

### B. KMeans K Sweep (K=3,5,7,10)
- HDBSCAN의 불균형 해소, 최적 K 자동 선정

### C. Soft Ensemble (Top-2 weighted blending)
- predict_proba로 top-2 preset weighted average

## Build 결과

### Regression (Val control MAE)
- Train control MAE: 0.0226 (expressiveness 0.0232, dynamics_center 0.0221)
- **Val control MAE: 0.0671** (expressiveness 0.0676, dynamics_center 0.0665)

### KMeans Sweep

| K | Max cluster ratio | Nearest MAE (ceiling) | Val accuracy | Val pred MAE |
|---|---|---|---|---|
| 3 | 35.7% | 0.0647 | 64.8% | 0.0830 |
| 5 | 26.9% | 0.0520 | 50.6% | 0.0802 |
| 7 | 19.3% | 0.0445 | 41.3% | 0.0794 |
| **10** | **14.7%** | **0.0384** | **35.1%** | **0.0781** |

- Best K=10 (by val_pred_mae)
- KMeans K=3도 HDBSCAN K=3보다 균형적 (35.7% vs 83%)

## Eval 결과 (Test set, 177 pieces, 741,410 notes)

| Mode | MAE | macro MAE | MSE | CC | SD_ratio | Recall(10%) | Recall(5%) | Gap Closure |
|------|-----|-----------|-----|----|----------|-------------|------------|-------------|
| Oracle | 7.94 | — | — | — | — | — | — | 100% |
| **Regression** | **9.81** | **9.73** | **169.10** | **0.7170** | **85.3%** | **72.1%** | **42.9%** | **71.9%** |
| Soft preset (sweep K=10) | 10.15 | 10.12 | 181.12 | 0.7011 | 88.0% | 70.6% | 41.7% | 66.8% |
| Preset (sweep K=10) | 10.61 | 10.56 | 196.58 | 0.6823 | 90.9% | 68.5% | 40.0% | 59.9% |
| Soft preset (K=3) | 11.18 | 11.06 | 207.44 | 0.6630 | 76.2% | 65.0% | 36.5% | 51.4% |
| Preset (HDBSCAN K=3) | 11.68 | — | — | — | — | — | — | 43.8% |
| Default [0.5,0.5] | 14.60 | — | — | — | — | — | — | 0% |

## 분석

### Regression이 최선인 이유
1. **Discretization loss 제거가 가장 큰 단일 개선 요인**: 연속값 예측이 centroid 양자화보다 본질적으로 우월
2. **Val control MAE 0.0671 < 모든 discrete preset** (K=10 best도 0.0781)
3. **구현 단순**: regressor 파일 하나, presets.json/classifier 불필요
4. **모든 메트릭에서 1위**: MAE, CC, Recall 모두 최고

### K sweep 관찰
- K 증가 시 nearest-preset MAE(ceiling)는 계속 감소하나, classifier accuracy도 함께 하락
- K=10에서 val_pred_mae 0.0781로 수렴 — K를 더 올려도 classifier bottleneck으로 한계
- Soft ensemble이 hard assignment 대비 일관되게 개선 (K=10: 10.61→10.15, K=3: 11.68→11.18)

### Gap closure 달성
- 성공 기준 (gap closure > 50%, MAE < 11.27) **4개 방법 모두 달성**
- Regression 71.9% gap closure: oracle (7.94)까지 남은 gap은 1.87

## 결론

**Regression을 기본 control inference 방법으로 채택.**

잔여 gap (9.81 vs 7.94 = 1.87 MAE)의 원인:
- 21-dim window features의 정보 ceiling (velocity 자체를 보지 못함)
- Train→Test generalization gap (train control MAE 0.0226 vs val 0.0671)
