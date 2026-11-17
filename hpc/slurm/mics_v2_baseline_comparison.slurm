#!/bin/bash
#SBATCH --job-name=mics-v2-baselines
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%1
#SBATCH --output=mics-v2-baselines-%A_%a.out
#SBATCH --error=mics-v2-baselines-%A_%a.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
MANIFEST=${MICS_BASELINE_MANIFEST:-$PROJECT_ROOT/configs/mics_v2_single_seed_baselines_v1.json}
export MICS_DATA_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"
git diff --quiet
git diff --cached --quiet
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Source revision: $(git rev-parse HEAD)"
echo "Baseline task: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/validate_install.py
python scripts/run_single_seed_comparison.py run \
    --manifest "$MANIFEST" \
    --task-index "$SLURM_ARRAY_TASK_ID"
