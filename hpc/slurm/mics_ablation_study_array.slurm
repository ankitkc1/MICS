#!/bin/bash
#SBATCH --job-name=mics-ablation-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%1
#SBATCH --output=mics-ablation-full-%A_%a.out
#SBATCH --error=mics-ablation-full-%A_%a.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
DATASET_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}
BUNDLE_ROOT=${MICS_SPLIT_BUNDLE_ROOT:-$HOME/pmcs/split/generated/data_seed_20260905}
SCENARIO_CONFIG=${MICS_SCENARIO_CONFIG:-$BUNDLE_ROOT/balanced_calibration/materialized_config.json}
MANIFEST=${MICS_STUDY_MANIFEST:-$PROJECT_ROOT/configs/mics_ablation_study_v1.json}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$DATASET_ROOT"
test -f "$SCENARIO_CONFIG"
test -f "$MANIFEST"
git diff --quiet
git diff --cached --quiet

eval "$(python scripts/resolve_ablation_study.py \
    --manifest "$MANIFEST" \
    --task-index "$SLURM_ARRAY_TASK_ID" \
    --format shell)"

STUDY_ROOT=$PROJECT_ROOT/results/studies/$STUDY_ID
SEED_ROOT=$STUDY_ROOT/$SEED_ID
TRACE=$SEED_ROOT/system_trace.json
REPORT=$SEED_ROOT/ablation_matrix.json
IMPORTANCE_OUTPUT=${MICS_IMPORTANCE_OUTPUT:-$STUDY_ROOT/calibration/modality_importance.json}
FULL_ID=$SEED_ID-mics-impact
CORE_ID=$SEED_ID-mics-core
NO_QUEUE_ID=$SEED_ID-mics-no-queue
MODULAR_ID=$SEED_ID-mics-modular
UNIFORM_ID=$SEED_ID-mics-uniform-impact
FULL_DIR=$SEED_ROOT/$FULL_ID
CORE_DIR=$SEED_ROOT/$CORE_ID
NO_QUEUE_DIR=$SEED_ROOT/$NO_QUEUE_ID
MODULAR_DIR=$SEED_ROOT/$MODULAR_ID
UNIFORM_DIR=$SEED_ROOT/$UNIFORM_ID

test -f "$IMPORTANCE_OUTPUT"
mkdir -p "$SEED_ROOT"
if [[ ! -f "$TRACE" ]]; then
    python scripts/generate_system_trace.py \
        --dataset-root "$DATASET_ROOT" \
        --scenario-config "$SCENARIO_CONFIG" \
        --output "$TRACE" \
        --trace-id "$STUDY_ID-$SEED_ID" \
        --rounds "$ROUNDS" \
        --seed "$TRACE_SEED" \
        --minimum-available-clients "$MINIMUM_AVAILABLE_CLIENTS" \
        --availability-probability "$AVAILABILITY_PROBABILITY" \
        --completion-probability "$COMPLETION_PROBABILITY"
fi

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Source revision: $(git rev-parse HEAD)"
echo "Study: $STUDY_ID"
echo "Seed: $SEED_ID"
echo "Protocol SHA-256: $PROTOCOL_SHA256"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/validate_install.py

read -r PATCH_D PATCH_H PATCH_W <<< "$PATCH_SIZE"
read -r ROI_D ROI_H ROI_W <<< "$ROI_SIZE"
COMMON_ARGUMENTS=(
    --dataset-root "$DATASET_ROOT"
    --scenario-config "$SCENARIO_CONFIG"
    --system-trace "$TRACE"
    --output-root "$SEED_ROOT"
    --study-id "$STUDY_ID"
    --protocol-manifest "$PROTOCOL_PATH"
    --rounds "$ROUNDS"
    --clients-per-round "$CLIENTS_PER_ROUND"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --service-type "$SERVICE_TYPE"
    --saturation-level "$SATURATION_LEVEL"
    --safety-margin "$SAFETY_MARGIN"
    --reliability-aware
    --base-channels "$BASE_CHANNELS"
    --num-levels "$NUM_LEVELS"
    --fusion "$FUSION"
    --modality-dropout "$MODALITY_DROPOUT"
    --patch-size "$PATCH_D" "$PATCH_H" "$PATCH_W"
    --foreground-probability "$FOREGROUND_PROBABILITY"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --local-epochs "$LOCAL_EPOCHS"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --gradient-clip-norm "$GRADIENT_CLIP_NORM"
    --amp-initial-scale "$AMP_INITIAL_SCALE"
    --amp-max-retries "$AMP_MAX_RETRIES"
    --model-seed "$MODEL_SEED"
    --selection-seed "$SELECTION_SEED"
    --data-seed "$DATA_SEED"
    --device cuda
)

run_variant() {
    local run_id=$1
    local run_dir=$2
    shift 2
    local resume_argument=()
    if [[ -d "$run_dir" ]]; then
        python - "$run_dir/run_metadata.json" "$(git rev-parse HEAD)" <<'PY'
import json
import pathlib
import sys

metadata = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert metadata["source_revision"] == sys.argv[2], (
    "Refusing to resume or reuse a run under a different source revision."
)
assert metadata["source_dirty"] is False
PY
        if grep -q '"state": "completed"' "$run_dir/status.json"; then
            echo "Training already completed: $run_dir"
            return
        fi
        resume_argument=(--resume)
    fi
    python scripts/run_mics_training.py \
        --run-id "$run_id" \
        "${COMMON_ARGUMENTS[@]}" \
        "$@" \
        "${resume_argument[@]}"
}

run_variant "$FULL_ID" "$FULL_DIR" \
    --importance-output "$IMPORTANCE_OUTPUT" --impact-weight "$IMPACT_WEIGHT"
run_variant "$CORE_ID" "$CORE_DIR" --impact-weight 0
run_variant "$NO_QUEUE_ID" "$NO_QUEUE_DIR" \
    --importance-output "$IMPORTANCE_OUTPUT" --impact-weight "$IMPACT_WEIGHT" \
    --ablation no_component_queue
run_variant "$MODULAR_ID" "$MODULAR_DIR" \
    --importance-output "$IMPORTANCE_OUTPUT" --impact-weight "$IMPACT_WEIGHT" \
    --ablation modular_additive
run_variant "$UNIFORM_ID" "$UNIFORM_DIR" \
    --impact-weight "$IMPACT_WEIGHT" --ablation uniform_importance

evaluate_run() {
    local run_dir=$1
    local checkpoint
    for checkpoint in "$run_dir"/checkpoints/after_round_*.pt; do
        local stem=${checkpoint##*/after_round_}
        local round=${stem%.pt}
        local output=$run_dir/evaluations/validation_after_round_${round}.json
        if [[ -f "$output" ]]; then
            echo "Evaluation already completed: $output"
            continue
        fi
        python scripts/evaluate_brats_checkpoint.py \
            --run-dir "$run_dir" \
            --dataset-root "$DATASET_ROOT" \
            --scenario-config "$SCENARIO_CONFIG" \
            --checkpoint "$checkpoint" \
            --partition "$EVALUATION_PARTITION" \
            --roi-size "$ROI_D" "$ROI_H" "$ROI_W" \
            --overlap "$OVERLAP" \
            --device cuda \
            --output "$output"
    done
}

evaluate_run "$FULL_DIR"
evaluate_run "$CORE_DIR"
evaluate_run "$NO_QUEUE_DIR"
evaluate_run "$MODULAR_DIR"
evaluate_run "$UNIFORM_DIR"

FINAL_ROUND=$(printf '%04d' "$ROUNDS")
FULL_EVALUATION=$FULL_DIR/evaluations/validation_after_round_${FINAL_ROUND}.json
CORE_EVALUATION=$CORE_DIR/evaluations/validation_after_round_${FINAL_ROUND}.json
NO_QUEUE_EVALUATION=$NO_QUEUE_DIR/evaluations/validation_after_round_${FINAL_ROUND}.json
MODULAR_EVALUATION=$MODULAR_DIR/evaluations/validation_after_round_${FINAL_ROUND}.json
UNIFORM_EVALUATION=$UNIFORM_DIR/evaluations/validation_after_round_${FINAL_ROUND}.json

if [[ ! -f "$REPORT" ]]; then
    python scripts/compare_method_matrix.py \
        --method mics_impact "$FULL_DIR" "$FULL_EVALUATION" \
        --method mics_core "$CORE_DIR" "$CORE_EVALUATION" \
        --method mics_impact_no_queue "$NO_QUEUE_DIR" "$NO_QUEUE_EVALUATION" \
        --method mics_impact_modular "$MODULAR_DIR" "$MODULAR_EVALUATION" \
        --method mics_impact_uniform "$UNIFORM_DIR" "$UNIFORM_EVALUATION" \
        --reference-label mics_impact \
        --output "$REPORT"
fi

python - "$REPORT" "$PROTOCOL_SHA256" "$SEED_ID" "$STUDY_ID" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert report["purpose"] == "matched_completed_method_matrix_comparison"
assert report["interpretation_allowed"] is True
assert report["reference_label"] == "mics_impact"
for method in report["methods"].values():
    config = json.loads(
        (pathlib.Path(method["run_dir"]) / "resolved_config.json").read_text()
    )
    assert config["study"]["protocol_sha256"] == sys.argv[2]
    assert config["study"]["id"] == sys.argv[4]
    assert pathlib.Path(method["run_dir"]).name.startswith(sys.argv[3] + "-")
PY

echo "Seed report: $REPORT"
echo "MICS FULL ABLATION SEED TASK PASSED: $SEED_ID"
