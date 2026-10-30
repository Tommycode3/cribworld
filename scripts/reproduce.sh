#!/usr/bin/env bash
# Full pipeline, one process at a time. Usage: bash scripts/reproduce.sh [configs/default.yaml]
set -euo pipefail
CFG=${1:-configs/default.yaml}
export MUJOCO_GL=${MUJOCO_GL:-$([[ "$(uname)" == "Darwin" ]] && echo cgl || echo egl)}
export PYTHONUNBUFFERED=1
python scripts/freeplay.py   --config "$CFG"
python scripts/heldout.py    --config "$CFG"
python scripts/evaluate.py   --config "$CFG"
python scripts/downstream.py --config "$CFG"
python scripts/figures.py    --config "$CFG"
python scripts/gif.py        --config "$CFG"
