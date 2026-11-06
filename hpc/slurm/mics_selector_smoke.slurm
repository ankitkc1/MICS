#!/bin/bash
#SBATCH --job-name=mics-selector
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=mics-selector-%j.out
#SBATCH --error=mics-selector-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
DATASET_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}
BUNDLE_ROOT=${MICS_SPLIT_BUNDLE_ROOT:-$HOME/pmcs/split/generated/data_seed_20260905}
SCENARIO_CONFIG=${MICS_SCENARIO_CONFIG:-$BUNDLE_ROOT/balanced/materialized_config.json}
IMPORTANCE_OUTPUT=${MICS_IMPORTANCE_OUTPUT:?Set MICS_IMPORTANCE_OUTPUT to the completed importance JSON}
OUTPUT=${MICS_SELECTOR_OUTPUT:-$PROJECT_ROOT/results/mics-selector-smoke-${SLURM_JOB_ID}.json}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$DATASET_ROOT"
test -f "$SCENARIO_CONFIG"
test -f "$IMPORTANCE_OUTPUT"
test ! -e "$OUTPUT"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Source revision: $(git rev-parse HEAD)"
python scripts/validate_install.py

python scripts/smoke_mics_selector.py \
    --dataset-root "$DATASET_ROOT" \
    --scenario-config "$SCENARIO_CONFIG" \
    --importance-output "$IMPORTANCE_OUTPUT" \
    --output "$OUTPUT" \
    --rounds 20 \
    --clients-per-round 2 \
    --seed 202 \
    --impact-weight 1.0 \
    --safety-margin 0.1

test -f "$OUTPUT"
grep -q '"purpose": "selector_integration_smoke_not_model_quality"' "$OUTPUT"
grep -q '"test_partition_used_for_selection": false' "$OUTPUT"
echo "Selector output: $OUTPUT"
echo "MICS SELECTOR INTEGRATION JOB PASSED"
