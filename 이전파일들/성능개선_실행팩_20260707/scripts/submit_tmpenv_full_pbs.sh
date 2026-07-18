#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gnn_surrogate_2026}"
DATA_DIR="${DATA_DIR:-$HOME/04_npj/coarse_dataset}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/in_pi_mgn_tmpenv_full}"
JOB_NAME="${JOB_NAME:-gnn_tmpenv}"
QUEUE="${QUEUE:-coss_agpu}"
SELECT="${SELECT:-select=1:ncpus=6:mem=192g:ngpus=1:Qlist=agpu:container_engine=singularity}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-147.46.121.38:5000/ubuntu:18.04-gpu}"

export PATH="/opt/pbs/bin:/tools/scripts:${PATH:-}"
mkdir -p "$OUT_DIR"
JOB_FILE="$OUT_DIR/${JOB_NAME}.pbs"

cat > "$JOB_FILE" <<EOF
#PBS -N $JOB_NAME
#PBS -q $QUEUE
#PBS -l $SELECT
#PBS -j oe
#PBS -o $OUT_DIR/pbs.log

set -euo pipefail

echo "__JOB_START__ \$(date)"
echo "HOST=\$(hostname)"
nvidia-smi || true

ROOT="$ROOT"
DATA_DIR="$DATA_DIR"
OUT_DIR="$OUT_DIR"
TMP_ROOT="/tmp/rintern07_gnn_\${PBS_JOBID:-manual}"
VENV="\$TMP_ROOT/venv"

rm -rf "\$TMP_ROOT"
mkdir -p "\$TMP_ROOT"

/tools/anaconda3/bin/python -m venv "\$VENV"
source "\$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -i https://pypi.org/simple "numpy<2" "h5py==3.10.0" tqdm
python -m pip install torch==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118

python - <<'PY'
import h5py, torch
print("h5py", h5py.__version__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA")
if not torch.cuda.is_available():
    raise SystemExit("CUDA_NOT_AVAILABLE")
PY

cd "\$ROOT"
export PYTHONPATH="\$ROOT:\${PYTHONPATH:-}"

python scripts/train.py \\
  --data-dir "\$DATA_DIR" \\
  --output-dir "\$OUT_DIR" \\
  --model-variant in-pi-mgn \\
  --epochs 20 \\
  --message-passing-steps 15 \\
  --latent-dim 128 \\
  --hidden-dim 128 \\
  --lr 1e-4 \\
  --lr-min 1e-7 \\
  --lr-decay-start-epoch 16 \\
  --noise-std 0.003 \\
  --stats-samples 128 \\
  --eval-every 1 \\
  --device cuda

python scripts/evaluate.py \\
  --data-dir "\$DATA_DIR" \\
  --checkpoint "\$OUT_DIR/best.pt" \\
  --rollout-steps 50 \\
  --device cuda | tee "\$OUT_DIR/eval_50rmse.log"

echo "__JOB_DONE__ \$(date)"
EOF

echo "Submitting $JOB_FILE"
qsub -v CONTAINER_IMAGE="$CONTAINER_IMAGE",HOME="$HOME" "$JOB_FILE"
