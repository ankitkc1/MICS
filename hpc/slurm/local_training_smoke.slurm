#!/bin/bash
#SBATCH --job-name=mics-local-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=mics-local-train-%j.out
#SBATCH --error=mics-local-train-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
DATASET_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}
SPLIT_ROOT=${MICS_SPLIT_ROOT:-$HOME/pmcs/split/20_c8_heter_modalnum}
BASE_CHANNELS=${MICS_BASE_CHANNELS:-16}
MAX_BATCHES=${MICS_SMOKE_BATCHES:-2}

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

python scripts/smoke_local_training.py \
    --dataset-root "$DATASET_ROOT" \
    --split-root "$SPLIT_ROOT" \
    --client C1 \
    --modalities flair,t1ce,t1,t2 \
    --base-channels "$BASE_CHANNELS" \
    --max-batches "$MAX_BATCHES"

python scripts/smoke_local_training.py \
    --dataset-root "$DATASET_ROOT" \
    --split-root "$SPLIT_ROOT" \
    --client C6 \
    --modalities t1ce \
    --base-channels "$BASE_CHANNELS" \
    --max-batches "$MAX_BATCHES"

echo "MICS LOCAL-TRAINING GPU JOB PASSED"
