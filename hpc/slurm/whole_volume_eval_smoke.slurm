#!/bin/bash
#SBATCH --job-name=mics-whole-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=mics-whole-eval-%j.out
#SBATCH --error=mics-whole-eval-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
DATASET_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}
SPLIT_ROOT=${MICS_SPLIT_ROOT:-$HOME/pmcs/split/20_c8_heter_modalnum}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$DATASET_ROOT"
test -d "$SPLIT_ROOT"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Source revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/validate_install.py
python scripts/smoke_whole_volume_eval.py \
    --dataset-root "$DATASET_ROOT" \
    --split-root "$SPLIT_ROOT" \
    --client C1 \
    --sample-index 0 \
    --base-channels 16 \
    --overlap 0.5

echo "MICS WHOLE-VOLUME EVALUATION GPU JOB PASSED"
