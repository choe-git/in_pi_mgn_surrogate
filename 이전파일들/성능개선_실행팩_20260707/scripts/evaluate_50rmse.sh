#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gnn_surrogate_2026}"
DATA_DIR="${DATA_DIR:-$HOME/04_npj/coarse_dataset}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/best.pt" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

"$PYTHON_BIN" scripts/evaluate.py \
  --data-dir "$DATA_DIR" \
  --checkpoint "$1" \
  --rollout-steps 50 \
  --device "$DEVICE"
