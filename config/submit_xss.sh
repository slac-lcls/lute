#!/bin/bash
#SBATCH --account=lcls:mfxx1001621      # change to your account
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=8                       # number of MPI ranks; tune as needed
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=lute_xss_thermo
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUTE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-$SCRIPT_DIR/mfx101262725.yaml}"

cd "$LUTE_DIR"
srun python run_task.py \
    -t SmallDataXSSAnalyzer \
    -c "$CONFIG"
