#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gnn_surrogate_2026}"
DATA_DIR="${DATA_DIR:-$HOME/04_npj/coarse_dataset}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/in_pi_mgn_paper_like}"
JOB_NAME="${JOB_NAME:-gnn_surrogate}"
QUEUE="${QUEUE:-coss_agpu}"
SELECT="${SELECT:-select=1:ncpus=6:mem=192g:ngpus=1:Qlist=agpu:container_engine=singularity}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-147.46.121.38:5000/ubuntu:18.04-gpu}"
PYTHON_BIN="${PYTHON_BIN:-/tools/anaconda3/bin/python}"

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
cd "$ROOT"
export PYTHONPATH="$ROOT:\${PYTHONPATH:-}"
export DATA_DIR="$DATA_DIR"
export OUT_DIR="$OUT_DIR"
export PYTHON_BIN="$PYTHON_BIN"
export DEVICE="cuda"

bash scripts/run_paper_like.sh
EOF

echo "Submitting $JOB_FILE"
qsub -v CONTAINER_IMAGE="$CONTAINER_IMAGE",HOME="$HOME" "$JOB_FILE"
