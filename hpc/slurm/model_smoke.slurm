#!/bin/bash
#SBATCH --job-name=mics-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=mics-smoke-%j.out
#SBATCH --error=mics-smoke-%j.err

set -euo pipefail

PROJECT_ROOT=$(printenv MICS_PROJECT_ROOT || true)
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$HOME/MICS
fi

ENV_NAME=$(printenv MICS_CONDA_ENV || true)
if [ -z "$ENV_NAME" ]; then
    ENV_NAME=mics
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m pip install -e . --no-deps
python -m compileall -q src examples scripts tests
python scripts/validate_install.py
python -m unittest discover -s tests -v
python examples/brats_forward.py

echo "MICS HPC SMOKE TEST PASSED"
