#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gnn_surrogate_2026}"
DATA_DIR="${DATA_DIR:-$HOME/04_npj/coarse_dataset}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/stability_acceleration_rollout}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

"$PYTHON_BIN" scripts/train.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT_DIR" \
  --model-variant in-pi-mgn \
  --epochs 20 \
  --message-passing-steps 15 \
  --latent-dim 128 \
  --hidden-dim 128 \
  --lr 1e-4 \
  --lr-min 1e-7 \
  --lr-decay-start-epoch 16 \
  --noise-std 0.006 \
  --stats-samples all \
  --stats-sampling uniform \
  --output-scale acceleration \
  --acceleration-mode acceleration \
  --physics-operator gradient \
  --continuity-target zero \
  --data-loss-weight 0.5 \
  --continuity-weight 0.1666666667 \
  --convection-weight 0.1666666667 \
  --viscosity-weight 0.1666666667 \
  --eval-rollout-steps 50 \
  --best-rmse-steps 50 \
  --device "$DEVICE"
