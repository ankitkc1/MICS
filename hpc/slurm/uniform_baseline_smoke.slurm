#!/bin/bash
#SBATCH --job-name=mics-uniform
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=mics-uniform-%j.out
#SBATCH --error=mics-uniform-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
DATASET_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}
BUNDLE_ROOT=${MICS_SPLIT_BUNDLE_ROOT:-$HOME/pmcs/split/generated/data_seed_20260905}
SCENARIO_CONFIG=${MICS_SCENARIO_CONFIG:-$BUNDLE_ROOT/balanced/materialized_config.json}
RESULTS_ROOT=${MICS_RESULTS_ROOT:-$PROJECT_ROOT/results}
MAX_BATCHES=${MICS_SMOKE_BATCHES:-2}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$DATASET_ROOT"
test -f "$SCENARIO_CONFIG"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

RUN_ID="uniform-balanced-smoke-${SLURM_JOB_ID}"

echo "Source revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/validate_install.py

python scripts/run_uniform_baseline.py \
    --dataset-root "$DATASET_ROOT" \
    --scenario-config "$SCENARIO_CONFIG" \
    --output-root "$RESULTS_ROOT" \
    --run-id "$RUN_ID" \
    --rounds 2 \
    --clients-per-round 2 \
    --base-channels 16 \
    --max-batches-per-epoch "$MAX_BATCHES" \
    --model-seed 101 \
    --selection-seed 202 \
    --data-seed 303

test -f "$RESULTS_ROOT/$RUN_ID/checkpoints/last.pt"
grep -q '"state": "completed"' "$RESULTS_ROOT/$RUN_ID/status.json"
echo "MICS UNIFORM-RANDOM BASELINE SMOKE JOB PASSED"
