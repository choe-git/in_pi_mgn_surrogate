#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/gnn_surrogate_2026}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"

cd "$ROOT"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/inspect_dataset.py --help >/dev/null
echo "Environment ready at $ROOT/.venv"
