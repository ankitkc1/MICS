#!/bin/bash
#SBATCH --job-name=mics-ablation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=mics-ablation-%j.out
#SBATCH --error=mics-ablation-%j.err

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
TRACE=$RESULTS_ROOT/mics-ablation-trace-smoke-${SLURM_JOB_ID}.json
REPORT=$RESULTS_ROOT/mics-ablation-matrix-smoke-${SLURM_JOB_ID}.json
FULL_ID=mics-impact-ablation-smoke-${SLURM_JOB_ID}
CORE_ID=mics-core-ablation-smoke-${SLURM_JOB_ID}
NO_QUEUE_ID=mics-no-queue-ablation-smoke-${SLURM_JOB_ID}
MODULAR_ID=mics-modular-ablation-smoke-${SLURM_JOB_ID}
UNIFORM_ID=mics-uniform-impact-ablation-smoke-${SLURM_JOB_ID}
FULL_DIR=$RESULTS_ROOT/$FULL_ID
CORE_DIR=$RESULTS_ROOT/$CORE_ID
NO_QUEUE_DIR=$RESULTS_ROOT/$NO_QUEUE_ID
MODULAR_DIR=$RESULTS_ROOT/$MODULAR_ID
UNIFORM_DIR=$RESULTS_ROOT/$UNIFORM_ID
EVALUATION_NAME=validation_subset_ablation_job_${SLURM_JOB_ID}.json

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$DATASET_ROOT"
test -f "$SCENARIO_CONFIG"
test -f "$IMPORTANCE_OUTPUT"
test ! -e "$TRACE"
test ! -e "$REPORT"
test ! -e "$FULL_DIR"
test ! -e "$CORE_DIR"
test ! -e "$NO_QUEUE_DIR"
test ! -e "$MODULAR_DIR"
test ! -e "$UNIFORM_DIR"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Source revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/validate_install.py

python scripts/generate_system_trace.py \
    --dataset-root "$DATASET_ROOT" \
    --scenario-config "$SCENARIO_CONFIG" \
    --output "$TRACE" \
    --trace-id mics-ablation-neutral-smoke-${SLURM_JOB_ID} \
    --rounds "$ROUNDS" \
    --seed 606 \
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
    --service-type pure_coverage
    --saturation-level 1
    --safety-margin 0.1
    --reliability-aware
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

python scripts/run_mics_training.py \
    --run-id "$FULL_ID" \
    --importance-output "$IMPORTANCE_OUTPUT" \
    --impact-weight 1.0 \
    "${COMMON_ARGUMENTS[@]}"

python scripts/run_mics_training.py \
    --run-id "$CORE_ID" \
    --impact-weight 0.0 \
    "${COMMON_ARGUMENTS[@]}"

python scripts/run_mics_training.py \
    --run-id "$NO_QUEUE_ID" \
    --importance-output "$IMPORTANCE_OUTPUT" \
    --impact-weight 1.0 \
    --ablation no_component_queue \
    "${COMMON_ARGUMENTS[@]}"

python scripts/run_mics_training.py \
    --run-id "$MODULAR_ID" \
    --importance-output "$IMPORTANCE_OUTPUT" \
    --impact-weight 1.0 \
    --ablation modular_additive \
    "${COMMON_ARGUMENTS[@]}"

python scripts/run_mics_training.py \
    --run-id "$UNIFORM_ID" \
    --impact-weight 1.0 \
    --ablation uniform_importance \
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

evaluate_run "$FULL_DIR"
evaluate_run "$CORE_DIR"
evaluate_run "$NO_QUEUE_DIR"
evaluate_run "$MODULAR_DIR"
evaluate_run "$UNIFORM_DIR"

FULL_EVALUATION=$FULL_DIR/evaluations/$EVALUATION_NAME
CORE_EVALUATION=$CORE_DIR/evaluations/$EVALUATION_NAME
NO_QUEUE_EVALUATION=$NO_QUEUE_DIR/evaluations/$EVALUATION_NAME
MODULAR_EVALUATION=$MODULAR_DIR/evaluations/$EVALUATION_NAME
UNIFORM_EVALUATION=$UNIFORM_DIR/evaluations/$EVALUATION_NAME

python scripts/compare_method_matrix.py \
    --method mics_impact "$FULL_DIR" "$FULL_EVALUATION" \
    --method mics_core "$CORE_DIR" "$CORE_EVALUATION" \
    --method mics_impact_no_queue "$NO_QUEUE_DIR" "$NO_QUEUE_EVALUATION" \
    --method mics_impact_modular "$MODULAR_DIR" "$MODULAR_EVALUATION" \
    --method mics_impact_uniform "$UNIFORM_DIR" "$UNIFORM_EVALUATION" \
    --reference-label mics_impact \
    --output "$REPORT"

python - "$REPORT" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = [
    "mics_impact",
    "mics_core",
    "mics_impact_no_queue",
    "mics_impact_modular",
    "mics_impact_uniform",
]
assert report["method_order"] == expected
assert report["reference_label"] == "mics_impact"
assert report["purpose"] == (
    "matched_method_matrix_integration_audit_not_model_quality"
)
assert report["interpretation_allowed"] is False
methods = report["methods"]
assert methods["mics_impact"]["method"] == "mics_impact"
assert methods["mics_core"]["method"] == "mics_core"
assert methods["mics_impact_no_queue"]["method"] == (
    "mics_impact_no_queue"
)
assert methods["mics_impact_modular"]["method"] == "mics_impact_modular"
assert methods["mics_impact_uniform"]["method"] == "mics_impact_uniform"
assert methods["mics_impact"]["selection_configuration"][
    "component_queue_weight"
] == 1.0
assert methods["mics_impact_no_queue"]["selection_configuration"][
    "component_queue_weight"
] == 0.0
assert methods["mics_impact_modular"]["selection_configuration"][
    "subset_utility"
] == "modular_additive"
assert methods["mics_impact_uniform"]["selection_configuration"][
    "importance_mode"
] == "uniform"
full_source = methods["mics_impact"]["external_importance_source"]
assert full_source["test_data_accessed"] is False
assert full_source["probe_cost"] is not None
assert methods["mics_core"]["external_importance_source"] is None
assert methods["mics_impact_uniform"]["external_importance_source"] is None
PY

echo "System trace: $TRACE"
echo "Full MICS-Impact run: $FULL_DIR"
echo "MICS-Core run: $CORE_DIR"
echo "No-component-queue run: $NO_QUEUE_DIR"
echo "Modular-utility run: $MODULAR_DIR"
echo "Uniform-importance run: $UNIFORM_DIR"
echo "Ablation report: $REPORT"
echo "MICS MATCHED ABLATION-MATRIX PIPELINE JOB PASSED"
