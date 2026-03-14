#!/bin/bash
set -e

cd /home/tikim/repo/midi-velocity-infer-v2/v3

TRAIN=/home/tikim/dataset/maestro/maestro-raw/maestro-midi/train
VAL=/home/tikim/dataset/maestro/maestro-raw/maestro-midi/validation
TEST=/home/tikim/dataset/maestro/maestro-raw/maestro-midi/test
OUT=runs_ablation

# Control-enabled models: build_presets + 3 evals each
for RUN in no_beta no_dropout no_jitter no_reg cls_head stoch_head; do
  echo "=== Evaluating $RUN ==="

  # Build control regressor
  uv run python -m mvi_v3.cli.build_presets \
    --train-dir $TRAIN --val-dir $VAL \
    --stats $OUT/$RUN/stats.json \
    --output-dir $OUT/$RUN/presets_regress --method regress

  # Eval: oracle
  uv run python -m mvi_v3.cli.eval_baseline \
    --data-dir $TEST \
    --checkpoint $OUT/$RUN/best.pt \
    --stats $OUT/$RUN/stats.json \
    --control-mode oracle \
    --output-dir $OUT/$RUN/eval_oracle

  # Eval: regression
  uv run python -m mvi_v3.cli.eval_baseline \
    --data-dir $TEST \
    --checkpoint $OUT/$RUN/best.pt \
    --stats $OUT/$RUN/stats.json \
    --control-mode regression \
    --preset-dir $OUT/$RUN/presets_regress \
    --output-dir $OUT/$RUN/eval_regression

  # Eval: default
  uv run python -m mvi_v3.cli.eval_baseline \
    --data-dir $TEST \
    --checkpoint $OUT/$RUN/best.pt \
    --stats $OUT/$RUN/stats.json \
    --control-mode default \
    --output-dir $OUT/$RUN/eval_default
done

# cls_head additional: argmax decode
echo "=== cls_head argmax eval ==="
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST \
  --checkpoint $OUT/cls_head/best.pt \
  --stats $OUT/cls_head/stats.json \
  --control-mode oracle --decode-mode argmax \
  --output-dir $OUT/cls_head/eval_oracle_argmax

# No-control model: standard eval only
echo "=== Evaluating no_controls ==="
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir $TEST \
  --checkpoint $OUT/no_controls/best.pt \
  --stats $OUT/no_controls/stats.json \
  --output-dir $OUT/no_controls/eval_standard

echo "=== All ablation evaluations complete ==="

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
for RUN in no_controls no_beta no_dropout no_jitter no_reg cls_head stoch_head; do
  echo "--- $RUN ---"
  for EVAL_DIR in $OUT/$RUN/eval_*; do
    MODE=$(basename $EVAL_DIR)
    uv run python -c "
import json
d=json.load(open('$EVAL_DIR/metrics.json'))
a=d['aggregate']
print(f'  $MODE: MAE={a[\"weighted_mae\"]:.2f} CC={a[\"weighted_cc\"]:.3f} SD={a[\"weighted_sd_ratio\"]*100:.1f}% R10={a[\"weighted_recall_10\"]*100:.1f}%')
" 2>/dev/null || echo "  $MODE: FAILED"
  done
done
