#!/bin/bash
#SBATCH --job-name=mics-four-way
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=mics-four-way-%j.out
#SBATCH --error=mics-four-way-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
DATASET_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}
BUNDLE_ROOT=${MICS_SPLIT_BUNDLE_ROOT:-$HOME/pmcs/split/generated/data_seed_20260905}
SCENARIO_CONFIG=${MICS_SCENARIO_CONFIG:-$BUNDLE_ROOT/balanced/materialized_config.json}
IMPORTANCE_OUTPUT=${MICS_IMPORTANCE_OUTPUT:?Set MICS_IMPORTANCE_OUTPUT to the completed importance JSON}
RESULTS_ROOT=${MICS_RESULTS_ROOT:-$PROJECT_ROOT/results}
ROUNDS=${MICS_SMOKE_ROUNDS:-4}
BASE_CHANNELS=${MICS_SMOKE_BASE_CHANNELS:-8}
TRACE=$RESULTS_ROOT/four-way-trace-smoke-${SLURM_JOB_ID}.json
REPORT=$RESULTS_ROOT/four-way-method-matrix-smoke-${SLURM_JOB_ID}.json
UNIFORM_ID=uniform-four-way-smoke-${SLURM_JOB_ID}
POWER_ID=power-choice-four-way-smoke-${SLURM_JOB_ID}
DIVFL_ID=divfl-four-way-smoke-${SLURM_JOB_ID}
MICS_ID=mics-four-way-smoke-${SLURM_JOB_ID}
UNIFORM_DIR=$RESULTS_ROOT/$UNIFORM_ID
POWER_DIR=$RESULTS_ROOT/$POWER_ID
DIVFL_DIR=$RESULTS_ROOT/$DIVFL_ID
MICS_DIR=$RESULTS_ROOT/$MICS_ID
EVALUATION_NAME=validation_subset_four_way_job_${SLURM_JOB_ID}.json

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$DATASET_ROOT"
test -f "$SCENARIO_CONFIG"
test -f "$IMPORTANCE_OUTPUT"
test ! -e "$TRACE"
test ! -e "$REPORT"
test ! -e "$UNIFORM_DIR"
test ! -e "$POWER_DIR"
test ! -e "$DIVFL_DIR"
test ! -e "$MICS_DIR"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Source revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/validate_install.py

python scripts/generate_system_trace.py \
    --dataset-root "$DATASET_ROOT" \
    --scenario-config "$SCENARIO_CONFIG" \
    --output "$TRACE" \
    --trace-id four-method-neutral-smoke-${SLURM_JOB_ID} \
    --rounds "$ROUNDS" \
    --seed 505 \
    --minimum-available-clients 2 \
    --availability-probability 0.75 \
    --completion-probability 0.85

COMMON_ARGUMENTS=(
    --dataset-root "$DATASET_ROOT"
    --scenario-config "$SCENARIO_CONFIG"
    --system-trace "$TRACE"
    --output-root "$RESULTS_ROOT"
    --rounds "$ROUNDS"
    --clients-per-round 2
    --checkpoint-every 2
    --base-channels "$BASE_CHANNELS"
    --num-levels 4
    --fusion mean_var
    --modality-dropout 0.2
    --patch-size 96 96 96
    --foreground-probability 0.5
    --batch-size 1
    --num-workers 0
    --local-epochs 1
    --learning-rate 0.0001
    --weight-decay 0.00001
    --gradient-clip-norm 5.0
    --amp-initial-scale 4096
    --amp-max-retries 8
    --max-batches-per-epoch 1
    --model-seed 101
    --selection-seed 202
    --data-seed 303
    --device cuda
)

python scripts/run_uniform_baseline.py \
    --run-id "$UNIFORM_ID" \
    "${COMMON_ARGUMENTS[@]}"

python scripts/run_reference_baseline.py \
    --method power_of_choice \
    --run-id "$POWER_ID" \
    --candidate-size 4 \
    --probe-max-batches 1 \
    "${COMMON_ARGUMENTS[@]}"

python scripts/run_reference_baseline.py \
    --method divfl \
    --run-id "$DIVFL_ID" \
    --probe-max-batches 1 \
    "${COMMON_ARGUMENTS[@]}"

python scripts/run_mics_training.py \
    --run-id "$MICS_ID" \
    --importance-output "$IMPORTANCE_OUTPUT" \
    --service-type pure_coverage \
    --saturation-level 1 \
    --safety-margin 0.1 \
    --impact-weight 1.0 \
    --reliability-aware \
    "${COMMON_ARGUMENTS[@]}"

evaluate_run() {
    local run_dir=$1
    local output=$run_dir/evaluations/$EVALUATION_NAME
    python scripts/evaluate_brats_checkpoint.py \
        --run-dir "$run_dir" \
        --dataset-root "$DATASET_ROOT" \
        --scenario-config "$SCENARIO_CONFIG" \
        --partition validation \
        --max-cases-per-client 1 \
        --roi-size 96 96 96 \
        --overlap 0.5 \
        --device cuda \
        --output "$output"
}

evaluate_run "$UNIFORM_DIR"
evaluate_run "$POWER_DIR"
evaluate_run "$DIVFL_DIR"
evaluate_run "$MICS_DIR"

UNIFORM_EVALUATION=$UNIFORM_DIR/evaluations/$EVALUATION_NAME
POWER_EVALUATION=$POWER_DIR/evaluations/$EVALUATION_NAME
DIVFL_EVALUATION=$DIVFL_DIR/evaluations/$EVALUATION_NAME
MICS_EVALUATION=$MICS_DIR/evaluations/$EVALUATION_NAME

python scripts/compare_method_matrix.py \
    --method uniform_random "$UNIFORM_DIR" "$UNIFORM_EVALUATION" \
    --method power_of_choice "$POWER_DIR" "$POWER_EVALUATION" \
    --method divfl_gradient_facility "$DIVFL_DIR" "$DIVFL_EVALUATION" \
    --method mics_impact "$MICS_DIR" "$MICS_EVALUATION" \
    --reference-label uniform_random \
    --output "$REPORT"

python - "$REPORT" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = [
    "uniform_random",
    "power_of_choice",
    "divfl_gradient_facility",
    "mics_impact",
]
assert report["method_order"] == expected
assert report["purpose"] == (
    "matched_method_matrix_integration_audit_not_model_quality"
)
assert report["interpretation_allowed"] is False
methods = report["methods"]
assert methods["uniform_random"]["method"] == "uniform_random"
assert methods["power_of_choice"]["method"] == "power_of_choice"
assert methods["divfl_gradient_facility"]["method"] == (
    "divfl_gradient_facility"
)
assert methods["mics_impact"]["method"] == "mics_impact"
for label in ("power_of_choice", "divfl_gradient_facility"):
    selection = methods[label]["selection_and_training"]
    assert selection["cumulative_selection_probe_download_bytes"] > 0
    assert selection["cumulative_selection_probe_upload_bytes"] > 0
importance = methods["mics_impact"]["external_importance_source"]
assert importance["test_data_accessed"] is False
assert importance["probe_cost"] is not None
PY

echo "System trace: $TRACE"
echo "Uniform run: $UNIFORM_DIR"
echo "Power-of-Choice run: $POWER_DIR"
echo "DivFL-style run: $DIVFL_DIR"
echo "MICS run: $MICS_DIR"
echo "Method-matrix report: $REPORT"
echo "MICS MATCHED FOUR-METHOD PIPELINE JOB PASSED"
