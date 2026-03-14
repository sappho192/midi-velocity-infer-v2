#!/bin/bash
set -e

cd /home/tikim/repo/midi-velocity-infer-v2/v3

TRAIN=/home/tikim/dataset/maestro/maestro-raw/maestro-midi/train
VAL=/home/tikim/dataset/maestro/maestro-raw/maestro-midi/validation
BACKBONE=runs/ssl_pretrain/backbone.pt
OUT=runs_ablation
WB=mvi-v3-ablation

echo "=== [1/7] no_controls ==="
uv run python -m mvi_v3.cli.train_baseline \
  --train-dir $TRAIN --val-dir $VAL \
  --pretrained-backbone $BACKBONE \
  --dropout 0.2 --embedding-dropout 0.1 \
  --augment-velocity-jitter 2.0 \
  --velocity-weight-beta 3.0 \
  --patience 15 \
  --wandb-project $WB --wandb-run-name no_controls \
  --output-dir $OUT/no_controls

echo "=== [2/7] no_beta ==="
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

echo "=== [3/7] no_dropout ==="
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

echo "=== [4/7] no_jitter ==="
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

echo "=== [5/7] no_reg ==="
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

echo "=== [6/7] cls_head ==="
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

echo "=== [7/7] stoch_head ==="
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

echo "=== All ablation training complete ==="
