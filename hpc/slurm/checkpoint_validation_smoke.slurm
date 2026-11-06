#!/bin/bash
#SBATCH --job-name=mics-validation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=mics-validation-%j.out
#SBATCH --error=mics-validation-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
DATASET_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}
BUNDLE_ROOT=${MICS_SPLIT_BUNDLE_ROOT:-$HOME/pmcs/split/generated/data_seed_20260905}
SCENARIO_CONFIG=${MICS_SCENARIO_CONFIG:-$BUNDLE_ROOT/balanced/materialized_config.json}
RUN_DIR=${MICS_RUN_DIR:?Set MICS_RUN_DIR to the completed training run directory}
CHECKPOINT=${MICS_CHECKPOINT:-$RUN_DIR/checkpoints/last.pt}
OUTPUT=$RUN_DIR/evaluations/validation_subset_job_${SLURM_JOB_ID}.json

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$DATASET_ROOT"
test -f "$SCENARIO_CONFIG"
test -f "$CHECKPOINT"
test ! -e "$OUTPUT"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Source revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/validate_install.py

python scripts/evaluate_brats_checkpoint.py \
    --run-dir "$RUN_DIR" \
    --dataset-root "$DATASET_ROOT" \
    --scenario-config "$SCENARIO_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --partition validation \
    --max-cases-per-client 1 \
    --roi-size 96 96 96 \
    --overlap 0.5 \
    --output "$OUTPUT"

test -f "$OUTPUT"
grep -q '"purpose": "integration_smoke_not_model_quality"' "$OUTPUT"
echo "Evaluation output: $OUTPUT"
echo "MICS CHECKPOINT VALIDATION SMOKE JOB PASSED"
