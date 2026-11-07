#!/bin/bash
#SBATCH --job-name=mics-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=mics-train-%j.out
#SBATCH --error=mics-train-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
DATASET_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}
BUNDLE_ROOT=${MICS_SPLIT_BUNDLE_ROOT:-$HOME/pmcs/split/generated/data_seed_20260905}
SCENARIO_CONFIG=${MICS_SCENARIO_CONFIG:-$BUNDLE_ROOT/balanced/materialized_config.json}
IMPORTANCE_OUTPUT=${MICS_IMPORTANCE_OUTPUT:?Set MICS_IMPORTANCE_OUTPUT to the completed importance JSON}
RUN_ID=mics-impact-balanced-smoke-${SLURM_JOB_ID}
RUN_DIR=$PROJECT_ROOT/results/$RUN_ID

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$DATASET_ROOT"
test -f "$SCENARIO_CONFIG"
test -f "$IMPORTANCE_OUTPUT"
test ! -e "$RUN_DIR"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Source revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/validate_install.py

python scripts/run_mics_training.py \
    --dataset-root "$DATASET_ROOT" \
    --scenario-config "$SCENARIO_CONFIG" \
    --importance-output "$IMPORTANCE_OUTPUT" \
    --output-root "$PROJECT_ROOT/results" \
    --run-id "$RUN_ID" \
    --rounds 2 \
    --clients-per-round 2 \
    --checkpoint-every 1 \
    --service-type pure_coverage \
    --saturation-level 1 \
    --safety-margin 0.1 \
    --impact-weight 1.0 \
    --base-channels 16 \
    --num-levels 4 \
    --fusion mean_var \
    --modality-dropout 0.2 \
    --patch-size 96 96 96 \
    --batch-size 1 \
    --num-workers 0 \
    --local-epochs 1 \
    --max-batches-per-epoch 2 \
    --device cuda

test -f "$RUN_DIR/checkpoints/last.pt"
test -f "$RUN_DIR/checkpoints/after_round_0002.pt"
grep -q '"state": "completed"' "$RUN_DIR/status.json"
grep -q '"method": "mics_impact"' "$RUN_DIR/resolved_config.json"
echo "Run directory: $RUN_DIR"
echo "MICS TRACKED TRAINING SMOKE JOB PASSED"
