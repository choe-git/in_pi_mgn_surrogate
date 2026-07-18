#!/usr/bin/env bash
set -euo pipefail

LOG=/tmp/codex_setup_smoke.log
exec > >(tee -a "$LOG") 2>&1

echo "__SETUP_START__ $(date)"
rm -rf /tmp/codex_gnn_env
/tools/anaconda3/bin/python -m venv /tmp/codex_gnn_env
source /tmp/codex_gnn_env/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -i https://pypi.org/simple "numpy<2" "h5py==3.10.0" tqdm
python -m pip install torch==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118

python - <<'PY'
import h5py, torch
print('h5py', h5py.__version__)
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')
if not torch.cuda.is_available():
    raise SystemExit('CUDA_NOT_AVAILABLE')
PY

cd "$HOME/gnn_surrogate_2026"
PYTHONPATH=. /tmp/codex_gnn_env/bin/python scripts/train.py \
  --data-dir "$HOME/04_npj/coarse_dataset" \
  --output-dir runs/smoke_tmp_gpu \
  --model-variant in-pi-mgn \
  --epochs 1 \
  --message-passing-steps 1 \
  --latent-dim 16 \
  --hidden-dim 16 \
  --max-train-samples 1 \
  --max-test-cases 1 \
  --max-eval-steps 1 \
  --stats-samples 1 \
  --device cuda

echo "__SETUP_SMOKE_DONE__ $(date)"
