#!/bin/bash
# Interactive run — execute directly on a login/interactive node.
# Usage: bash run_xss.sh [config.yaml]
#   Default config: mfx101262725.yaml in this directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUTE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-$SCRIPT_DIR/mfx101262725.yaml}"
NPROCS="${SLURM_NPROCS:-2}"

cd "$LUTE_DIR"
SLURM_NPROCS=$NPROCS mpirun -n $NPROCS python run_task.py \
    -t SmallDataXSSAnalyzer \
    -c "$CONFIG"
