# Ablation Study Plan

## Context

v3의 최종 시스템은 여러 기법의 조합으로 구성됨: SSL pretraining, control conditioning, V-shaped loss, dropout 강화, velocity jitter augmentation, regression head. ISMIR 논문을 위해 각 구성요소의 기여도를 정량화하는 removal ablation 실험이 필요.

**현재 최고 성능** (ssl_finetune, regression control mode): MAE 9.81, CC 0.717, SD_ratio 85.3%

## Ablation Matrix

Baseline(full system)에서 하나씩 제거하는 방식. 총 9개 configuration (2개 재사용 + 7개 신규 학습).

| # | Run Name | 제거/변경 요소 | Controls | SSL | head | dropout | emb_drop | jitter | β |
|---|----------|---------------|----------|-----|------|---------|----------|--------|---|
| 0 | **full** (기존 ssl_finetune) | 없음 (기준) | Yes | Yes | reg | 0.2 | 0.1 | 2.0 | 3.0 |
| 1 | **no_ssl** (기존 phase5a) | SSL pretraining | Yes | **No** | reg | 0.2 | 0.1 | 2.0 | 3.0 |
| 2 | **no_controls** | Control conditioning | **No** | Yes | reg | 0.2 | 0.1 | 2.0 | 3.0 |
| 3 | **no_beta** | V-shaped loss | Yes | Yes | reg | 0.2 | 0.1 | 2.0 | **0.0** |
| 4 | **no_dropout** | 강화된 dropout | Yes | Yes | reg | **0.1** | **0.0** | 2.0 | 3.0 |
| 5 | **no_jitter** | Velocity jitter | Yes | Yes | reg | 0.2 | 0.1 | **0.0** | 3.0 |
| 6 | **no_reg** | 전체 정규화 | Yes | Yes | reg | **0.1** | **0.0** | **0.0** | 3.0 |
| 7 | **cls_head** | Head → classification | Yes | Yes | **cls** | 0.2 | 0.1 | 2.0 | 3.0 |
| 8 | **stoch_head** | Head → stochastic | Yes | Yes | **stoch** | 0.2 | 0.1 | 2.0 | 3.0 |

## 재사용 가능한 기존 모델

- **Run 0 (full)**: `v3/runs/ssl_finetune/` — config 완전 일치, eval 완료
- **Run 1 (no_ssl)**: `v3/runs/phase5a_oracle/` — SSL 없이 동일 config로 학습됨. Oracle eval 완료 (MAE 8.30). **Regression/default eval 추가 필요**

## 실행 계획

### Phase A: wandb 통합 코드 수정 ✅

파일: `v3/mvi_v3/cli/train_baseline.py`
- CLI 인자 2개 추가: `--wandb-project`, `--wandb-run-name`
- `main()` 내 wandb 초기화, epoch logging, finish 추가
- 변경 최소화: 기존 monitoring 시스템은 그대로 유지, wandb는 추가 logging만

### Phase B: no_ssl 추가 평가 (기존 checkpoint 활용, ~10분)

```bash
cd /home/tikim/repo/midi-velocity-infer-v2/v3

TRAIN=/home/tikim/dataset/maestro/maestro-raw/maestro-midi/train
VAL=/home/tikim/dataset/maestro/maestro-raw/maestro-midi/validation
TEST=/home/tikim/dataset/maestro/maestro-raw/maestro-midi/test

# 1. Build control regressor
uv run python -m mvi_v3.cli.build_presets \
  --train-dir $TRAIN --val-dir $VAL \
  --stats runs/phase5a_oracle/stats.json \
  --output-dir runs_ablation/no_ssl/presets_regress \
  --method regress

# 2. Eval: regression
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST \
  --checkpoint runs/phase5a_oracle/best.pt \
  --stats runs/phase5a_oracle/stats.json \
  --control-mode regression \
  --preset-dir runs_ablation/no_ssl/presets_regress \
  --output-dir runs_ablation/no_ssl/eval_regression

# 3. Eval: default
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST \
  --checkpoint runs/phase5a_oracle/best.pt \
  --stats runs/phase5a_oracle/stats.json \
  --control-mode default \
  --output-dir runs_ablation/no_ssl/eval_default

# 4. Eval: oracle
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST \
  --checkpoint runs/phase5a_oracle/best.pt \
  --stats runs/phase5a_oracle/stats.json \
  --control-mode oracle \
  --output-dir runs_ablation/no_ssl/eval_oracle
```

### Phase C: 신규 모델 학습 (7개, 각 1-2시간)

공통 인자:
```bash
BACKBONE=runs/ssl_pretrain/backbone.pt
OUT=runs_ablation
WB=mvi-v3-ablation  # wandb project name
```

**Run 2: no_controls**
```bash
uv run python -m mvi_v3.cli.train_baseline \
  --train-dir $TRAIN --val-dir $VAL \
  --pretrained-backbone $BACKBONE \
  --dropout 0.2 --embedding-dropout 0.1 \
  --augment-velocity-jitter 2.0 \
  --velocity-weight-beta 3.0 \
  --patience 15 \
  --wandb-project $WB --wandb-run-name no_controls \
  --output-dir $OUT/no_controls
```

**Run 3: no_beta**
```bash
uv run python -m mvi_v3.cli.train_baseline \
  --train-dir $TRAIN --val-dir $VAL \
  --pretrained-backbone $BACKBONE \
  --enable-controls --control-dims 2 \
  --dropout 0.2 --embedding-dropout 0.1 \
  --augment-velocity-jitter 2.0 \
  --velocity-weight-beta 0.0 \
  --patience 15 \
  --wandb-project $WB --wandb-run-name no_beta \
  --output-dir $OUT/no_beta
```

**Run 4: no_dropout**
```bash
uv run python -m mvi_v3.cli.train_baseline \
  --train-dir $TRAIN --val-dir $VAL \
  --pretrained-backbone $BACKBONE \
  --enable-controls --control-dims 2 \
  --dropout 0.1 --embedding-dropout 0.0 \
  --augment-velocity-jitter 2.0 \
  --velocity-weight-beta 3.0 \
  --patience 15 \
  --wandb-project $WB --wandb-run-name no_dropout \
  --output-dir $OUT/no_dropout
```

**Run 5: no_jitter**
```bash
uv run python -m mvi_v3.cli.train_baseline \
  --train-dir $TRAIN --val-dir $VAL \
  --pretrained-backbone $BACKBONE \
  --enable-controls --control-dims 2 \
  --dropout 0.2 --embedding-dropout 0.1 \
  --augment-velocity-jitter 0.0 \
  --velocity-weight-beta 3.0 \
  --patience 15 \
  --wandb-project $WB --wandb-run-name no_jitter \
  --output-dir $OUT/no_jitter
```

**Run 6: no_reg**
```bash
uv run python -m mvi_v3.cli.train_baseline \
  --train-dir $TRAIN --val-dir $VAL \
  --pretrained-backbone $BACKBONE \
  --enable-controls --control-dims 2 \
  --dropout 0.1 --embedding-dropout 0.0 \
  --augment-velocity-jitter 0.0 \
  --velocity-weight-beta 3.0 \
  --patience 15 \
  --wandb-project $WB --wandb-run-name no_reg \
  --output-dir $OUT/no_reg
```

**Run 7: cls_head** (classification, expectation decode)
```bash
uv run python -m mvi_v3.cli.train_baseline \
  --train-dir $TRAIN --val-dir $VAL \
  --pretrained-backbone $BACKBONE \
  --enable-controls --control-dims 2 \
  --head-type classification --label-smoothing 0.1 \
  --dropout 0.2 --embedding-dropout 0.1 \
  --augment-velocity-jitter 2.0 \
  --velocity-weight-beta 3.0 \
  --patience 15 \
  --wandb-project $WB --wandb-run-name cls_head \
  --output-dir $OUT/cls_head
```

**Run 8: stoch_head** (stochastic, Gaussian NLL)
```bash
uv run python -m mvi_v3.cli.train_baseline \
  --train-dir $TRAIN --val-dir $VAL \
  --pretrained-backbone $BACKBONE \
  --enable-controls --control-dims 2 \
  --head-type stochastic \
  --dropout 0.2 --embedding-dropout 0.1 \
  --augment-velocity-jitter 2.0 \
  --velocity-weight-beta 3.0 \
  --patience 15 \
  --wandb-project $WB --wandb-run-name stoch_head \
  --output-dir $OUT/stoch_head
```

### Phase D: 평가 (Phase C 완료 후)

Control-enabled 모델 (no_beta, no_dropout, no_jitter, no_reg, cls_head, stoch_head): 각각 build_presets + 3가지 eval (oracle/regression/default)

```bash
# 패턴 (각 run에 대해 반복):
RUN=no_beta  # no_dropout, no_jitter, no_reg, cls_head, stoch_head도 동일 패턴

uv run python -m mvi_v3.cli.build_presets \
  --train-dir $TRAIN --val-dir $VAL \
  --stats $OUT/$RUN/stats.json \
  --output-dir $OUT/$RUN/presets_regress --method regress

for MODE in oracle regression default; do
  EXTRA=""
  if [ "$MODE" = "regression" ]; then
    EXTRA="--preset-dir $OUT/$RUN/presets_regress"
  fi
  uv run python -m mvi_v3.cli.eval_baseline \
    --data-dir $TEST \
    --checkpoint $OUT/$RUN/best.pt \
    --stats $OUT/$RUN/stats.json \
    --control-mode $MODE $EXTRA \
    --output-dir $OUT/$RUN/eval_$MODE
done
```

cls_head eval 시 `--decode-mode expectation` (기본값) 사용. argmax도 추가로 실행:
```bash
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST \
  --checkpoint $OUT/cls_head/best.pt \
  --stats $OUT/cls_head/stats.json \
  --control-mode oracle --decode-mode argmax \
  --output-dir $OUT/cls_head/eval_oracle_argmax
```

No-control 모델 (no_controls): standard eval만
```bash
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST \
  --checkpoint $OUT/no_controls/best.pt \
  --stats $OUT/no_controls/stats.json \
  --output-dir $OUT/no_controls/eval_standard
```

### Phase E: 결과 수집 및 분석

모든 `eval_*/metrics.json`에서 weighted 메트릭 추출하여 ablation table 작성.

## 각 비교가 보여주는 것

| 비교 | ISMIR 논문 스토리 |
|------|-------------------|
| full vs no_ssl | SSL pretraining (GiantMIDI-Piano MNM)이 일반화에 기여하는 정도 |
| full vs no_controls | Control conditioning의 가치 — 없으면 piece-level style 정보 부재 |
| full vs no_beta | V-shaped loss (He 2025)가 극단 velocity 예측에 미치는 영향 |
| full vs no_dropout | 강화된 dropout (0.1→0.2) + embedding dropout의 과적합 억제 효과 |
| full vs no_jitter | Velocity jitter augmentation의 정규화 효과 |
| full vs no_reg | 정규화 전체 제거 — 개별 효과의 합 vs 실제 (interaction 분석) |
| full vs cls_head | Classification head의 장단점 (SD_ratio 향상 가능성, MAE tradeoff) |
| full vs stoch_head | Stochastic head의 확률적 예측 능력 평가 |

## 논문용 결과 테이블 템플릿

### Table 1: Component Ablation (regression head 기준)

```
| Configuration        | MAE(reg) | CC(reg) | SD_ratio(reg) | R10%(reg) | MAE(oracle) |
|----------------------|----------|---------|---------------|-----------|-------------|
| Full system          | 9.81     | 0.717   | 85.3%         | 72.1%     | 7.94        |
| − SSL pretraining    | ?        | ?       | ?             | ?         | 8.30        |
| − Control cond.      | ?(std)   | ?(std)  | ?(std)        | ?(std)    | N/A         |
| − V-shaped loss (β)  | ?        | ?       | ?             | ?         | ?           |
| − Extra dropout      | ?        | ?       | ?             | ?         | ?           |
| − Velocity jitter    | ?        | ?       | ?             | ?         | ?           |
| − All regularization | ?        | ?       | ?             | ?         | ?           |
| He et al. 2025       | 11.5*    | —       | —             | —         | —           |
| v2 baseline          | 13.87    | 0.344   | 36.1%         | 52.4%     | —           |
```

### Table 2: Head Type Comparison (SSL + controls + full regularization)

```
| Head Type            | MAE(reg) | CC(reg) | SD_ratio(reg) | R10%(reg) | MAE(oracle) |
|----------------------|----------|---------|---------------|-----------|-------------|
| Regression (β=3)     | 9.81     | 0.717   | 85.3%         | 72.1%     | 7.94        |
| Classification (exp) | ?        | ?       | ?             | ?         | ?           |
| Stochastic (NLL)     | ?        | ?       | ?             | ?         | ?           |
```

## 디렉토리 구조

```
v3/runs_ablation/
├── no_ssl/          (phase5a 재사용 + 추가 eval)
├── no_controls/     (신규 학습)
├── no_beta/         (신규 학습)
├── no_dropout/      (신규 학습)
├── no_jitter/       (신규 학습)
├── no_reg/          (신규 학습)
├── cls_head/        (신규 학습)
└── stoch_head/      (신규 학습)
```

## 검증

- wandb dashboard에서 모든 run의 train/val loss curve 비교 확인
- 각 학습 완료 후 `config.json`의 변경된 파라미터가 정확히 1개(또는 head type 관련 그룹)인지 확인
- eval 결과의 `n_pieces=177`, `n_notes=741410` 일치 확인
- 최종 ablation table에서 full system 수치가 기존 결과와 일치하는지 확인
