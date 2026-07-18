#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gnn_surrogate_2026}"
DATA_DIR="${DATA_DIR:-$HOME/04_npj/coarse_dataset}"
BASE_OUT_DIR="${BASE_OUT_DIR:-$ROOT/runs/stability_noise_sweep}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

for NOISE_STD in 0.003 0.006 0.01; do
  "$PYTHON_BIN" scripts/train.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$BASE_OUT_DIR/noise_${NOISE_STD}" \
    --model-variant in-mgn \
    --epochs 20 \
    --message-passing-steps 15 \
    --latent-dim 128 \
    --hidden-dim 128 \
    --lr 1e-4 \
    --lr-min 1e-7 \
    --lr-decay-start-epoch 16 \
    --noise-std "$NOISE_STD" \
    --stats-samples 512 \
    --stats-sampling uniform \
    --output-scale delta \
    --accel-mode delta \
    --eval-rollout-steps 50 \
    --selection-metric rollout \
    --eval-every 1 \
    --device "$DEVICE"
done
