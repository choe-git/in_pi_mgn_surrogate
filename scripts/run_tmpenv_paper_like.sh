#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gnn_surrogate_2026}"
DATA_DIR="${DATA_DIR:-$HOME/04_npj/coarse_dataset}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/in_pi_mgn_tmpenv_screen}"
VENV="${VENV:-/tmp/codex_gnn_env}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/screen_train.log"
exec > >(tee -a "$LOG") 2>&1

echo "__RUN_START__ $(date)"
echo "HOST=$(hostname)"
echo "ROOT=$ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "VENV=$VENV"
nvidia-smi || true

rm -f "$OUT_DIR/.running" "$OUT_DIR/.done" "$OUT_DIR/.failed"
touch "$OUT_DIR/.running"
finish() {
  status=$?
  rm -f "$OUT_DIR/.running"
  if [ "$status" -eq 0 ]; then
    touch "$OUT_DIR/.done"
  else
    touch "$OUT_DIR/.failed"
  fi
  echo "__RUN_EXIT__ status=$status $(date)"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT TERM

if [ ! -x "$VENV/bin/python" ]; then
  /tools/anaconda3/bin/python -m venv "$VENV"
fi
source "$VENV/bin/activate"

if ! python - <<'PY'
import h5py
import torch

assert h5py.__version__ == "3.10.0", h5py.__version__
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.cuda.is_available()
PY
then
  python -m pip install --upgrade pip
  python -m pip install --only-binary=:all: -i https://pypi.org/simple "numpy<2" "h5py==3.10.0" tqdm
  python -m pip install torch==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
fi

python - <<'PY'
import h5py
import torch

print("h5py", h5py.__version__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA")
if not torch.cuda.is_available():
    raise SystemExit("CUDA_NOT_AVAILABLE")
PY

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python scripts/train.py \
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
  --noise-std 0.003 \
  --stats-samples 128 \
  --device "$DEVICE"

python scripts/evaluate.py \
  --data-dir "$DATA_DIR" \
  --checkpoint "$OUT_DIR/best.pt" \
  --rollout-steps 50 \
  --device "$DEVICE" | tee "$OUT_DIR/eval_50rmse.log"

echo "__RUN_DONE__ $(date)"
