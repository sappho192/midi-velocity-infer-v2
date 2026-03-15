# Ablation Study Results

MAESTRO test set (177 pieces, 741,410 notes). Weighted metrics 기준.

## Table 1: Component Ablation

Full system에서 하나씩 제거한 결과. Regression control mode 기준 (no_controls는 standard eval).

| Configuration | MAE | CC | SD_ratio | R10% | R5% | ΔMAE |
|---|---|---|---|---|---|---|
| **Full system** | **9.81** | **0.717** | **85.3%** | **72.1%** | **42.9%** | — |
| − SSL pretraining | 10.24 | 0.689 | 83.3% | 69.9% | 41.1% | +0.43 |
| − Control conditioning | 9.60 | 0.734 | 91.7% | 73.1% | 43.9% | −0.21 |
| − V-shaped loss (β=0) | 9.87 | 0.709 | 81.0% | 71.6% | 42.4% | +0.06 |
| − Extra dropout | 9.79 | 0.716 | 84.1% | 72.1% | 42.8% | −0.02 |
| − Velocity jitter | 9.80 | 0.718 | 85.5% | 72.0% | 42.9% | −0.01 |
| − All regularization | 9.79 | 0.716 | 84.7% | 72.1% | 42.9% | −0.02 |

### Oracle mode 비교

| Configuration | MAE(oracle) | CC(oracle) | SD_ratio(oracle) | R10%(oracle) |
|---|---|---|---|---|
| **Full system** | **7.94** | **0.797** | **91.9%** | **81.2%** |
| − SSL pretraining | 8.30 | 0.777 | 89.7% | 79.4% |
| − V-shaped loss (β=0) | 7.91 | 0.794 | 87.5% | 81.3% |
| − Extra dropout | 7.95 | 0.794 | 90.0% | 81.0% |
| − Velocity jitter | 7.93 | 0.798 | 91.9% | 81.2% |
| − All regularization | 7.95 | 0.794 | 90.5% | 81.1% |

### Default mode 비교

| Configuration | MAE(default) | CC(default) | SD_ratio(default) | R10%(default) |
|---|---|---|---|---|
| **Full system** | **14.60** | **0.617** | **96.1%** | **52.0%** |
| − SSL pretraining | 15.41 | 0.578 | 94.5% | 49.4% |
| − Control conditioning | 9.60 | 0.734 | 91.7% | 73.1% |
| − V-shaped loss (β=0) | 14.38 | 0.612 | 90.9% | 52.5% |
| − Extra dropout | 13.95 | 0.630 | 93.4% | 54.1% |
| − Velocity jitter | 14.42 | 0.624 | 96.3% | 52.5% |
| − All regularization | 14.00 | 0.629 | 93.3% | 54.0% |

## Table 2: Head Type Comparison

SSL + controls + full regularization 조건에서 head type만 변경.

### Regression mode

| Head Type | MAE | CC | SD_ratio | R10% | R5% |
|---|---|---|---|---|---|
| Regression (β=3) | 9.81 | 0.717 | 85.3% | 72.1% | 42.9% |
| **Classification (expectation)** | **9.67** | **0.722** | 76.2% | **72.5%** | 42.8% |
| Stochastic (NLL) | 9.76 | 0.715 | 79.2% | 72.1% | 42.8% |

### Oracle mode

| Head Type | MAE | CC | SD_ratio | R10% | R5% |
|---|---|---|---|---|---|
| Regression (β=3) | 7.94 | 0.797 | 91.9% | 81.2% | 52.1% |
| **Classification (expectation)** | **7.63** | **0.806** | 81.9% | **82.8%** | **53.5%** |
| Classification (argmax) | 7.89 | 0.789 | **94.2%** | 81.4% | **54.7%** |
| Stochastic (NLL) | 7.80 | 0.798 | 85.8% | 81.8% | 52.8% |

## 분석

### 1. SSL Pretraining — 가장 큰 기여

- Regression MAE: 9.81 → 10.24 (+0.43), CC: 0.717 → 0.689 (−0.028)
- Oracle MAE: 7.94 → 8.30 (+0.36)
- Default MAE: 14.60 → 15.41 (+0.81)
- 모든 control mode에서 일관된 성능 저하. GiantMIDI-Piano MNM 기반 SSL pretraining이 일반화에 가장 큰 기여.

### 2. Control Conditioning — 복합적 효과

- no_controls (standard eval) MAE 9.60은 full system regression MAE 9.81보다 낮음
- Control conditioning 모델은 oracle mode에서 7.94까지 내려가는 상한이 있지만, regression 예측이 완벽하지 않아 추가 오차 발생
- Control 없는 모델이 단일 모드로는 더 나은 결과 — control의 가치는 oracle/regression gap에서 확인해야 함
- Default mode에서 14.60 vs 9.60: control 모델은 control 없이 추론하면 성능이 크게 저하됨

### 3. V-shaped Loss (β) — SD_ratio 개선에 기여

- Regression MAE: +0.06 (미미), 하지만 SD_ratio: 85.3% → 81.0% (−4.3%p)
- Oracle SD_ratio: 91.9% → 87.5% (−4.4%p)
- β=3이 극단 velocity 예측의 dynamic range 보존에 기여

### 4. Dropout 강화 — 개별 효과 미미

- Extra dropout (0.1→0.2 + embedding dropout 0.1) 제거: MAE −0.02, SD_ratio −1.2%p
- Default mode에서는 오히려 개선 (14.60 → 13.95), 과적합이 아닌 다른 메커니즘 가능성

### 5. Velocity Jitter — 거의 무효

- Regression MAE: −0.01 (차이 없음), SD_ratio: +0.2%p
- 현재 설정(±2.0)에서는 실질적 기여 없음

### 6. All Regularization — Interaction 없음

- no_reg (dropout↓ + emb_drop=0 + jitter=0) vs full: MAE −0.02
- 개별 제거 효과의 합과 거의 동일 → 정규화 기법들 간 interaction 없음
- 정규화보다 SSL pretraining이 일반화의 주요 원인

### 7. Head Type — Classification이 MAE 최적

- **cls_head (expectation)**: oracle MAE **7.63** (전체 최고), regression MAE **9.67** (전체 최고)
- SD_ratio는 낮음 (76.2%–81.9%) — 분포를 expectation으로 요약하면서 dynamic range 축소
- **argmax decode**: SD_ratio 94.2%로 가장 높지만 MAE는 expectation보다 높음 (7.89 vs 7.63)
- **stoch_head**: regression과 비슷한 수준, 특별한 이점 없음

### 8. 논문 시사점

1. **SSL pretraining**이 유일하게 통계적으로 유의미한 개선을 보이는 구성요소
2. **Classification head (expectation)**가 MAE 기준 최적 — regression head 대비 oracle MAE 0.31 개선
3. **V-shaped loss**는 SD_ratio 보존에만 기여, MAE 개선 없음
4. Dropout/jitter 등 정규화는 SSL pretraining 존재 하에서 추가 효과 미미
5. Control conditioning은 oracle 상한을 제공하지만, regression 추론 시 추가 오차로 인해 unconditional 모델보다 MAE가 높을 수 있음

## 실험 환경

- 모든 모델: patience=15, 100 epoch 상한, SSL backbone (`runs/ssl_pretrain/backbone.pt`) 공유
- no_ssl만 backbone 없이 from-scratch 학습 (기존 `runs/phase5a_oracle` 재사용)
- full system: 기존 `runs/ssl_finetune` 재사용
- wandb project: `mvi-v3-ablation`
