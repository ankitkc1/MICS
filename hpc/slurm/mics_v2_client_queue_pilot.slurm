#!/bin/bash
#SBATCH --job-name=mics-v2-pilot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=mics-v2-pilot-%j.out
#SBATCH --error=mics-v2-pilot-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
DATASET_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}
BUNDLE_ROOT=${MICS_SPLIT_BUNDLE_ROOT:-$HOME/pmcs/split/generated/data_seed_20260905}
SCENARIO_CONFIG=${MICS_SCENARIO_CONFIG:-$BUNDLE_ROOT/balanced_calibration/materialized_config.json}
SOURCE_STUDY=$PROJECT_ROOT/results/studies/mics_ablation_balanced_v1
IMPORTANCE_OUTPUT=${MICS_IMPORTANCE_OUTPUT:-$SOURCE_STUDY/calibration/modality_importance.json}
TRACE=${MICS_SYSTEM_TRACE:-$SOURCE_STUDY/s01/system_trace.json}
MANIFEST=$PROJECT_ROOT/configs/mics_v2_client_queue_pilot_v1.json
STUDY_ID=mics_v2_client_queue_pilot_v1
STUDY_ROOT=$PROJECT_ROOT/results/studies/$STUDY_ID
PREFLIGHT=$STUDY_ROOT/selector_preflight.json
REPORT=$STUDY_ROOT/mics_v2_pilot_matrix.json
FULL_ID=mics-v2-impact
CORE_ID=mics-v2-core
NO_CLIENT_ID=mics-v2-no-client-queue
FULL_DIR=$STUDY_ROOT/$FULL_ID
CORE_DIR=$STUDY_ROOT/$CORE_ID
NO_CLIENT_DIR=$STUDY_ROOT/$NO_CLIENT_ID

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$DATASET_ROOT"
test -f "$SCENARIO_CONFIG"
test -f "$IMPORTANCE_OUTPUT"
test -f "$TRACE"
test -f "$MANIFEST"
git diff --quiet
git diff --cached --quiet

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
SOURCE_REVISION=$(git rev-parse HEAD)

echo "Source revision: $SOURCE_REVISION"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/validate_install.py

mkdir -p "$STUDY_ROOT"
LOCK_FILE=$STUDY_ROOT/.mics_v2_pilot.lock
exec 9>>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another MICS-v2 pilot job is already writing to $STUDY_ROOT" >&2
    echo "Refusing concurrent execution to protect checkpoints and logs." >&2
    exit 75
fi
printf 'job_id=%s source_revision=%s acquired_utc=%s\n' \
    "${SLURM_JOB_ID:-unknown}" \
    "$SOURCE_REVISION" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&9

if [[ ! -f "$PREFLIGHT" ]]; then
    python scripts/simulate_mics_v2_selector.py \
        --dataset-root "$DATASET_ROOT" \
        --scenario-config "$SCENARIO_CONFIG" \
        --system-trace "$TRACE" \
        --importance-output "$IMPORTANCE_OUTPUT" \
        --budget 2 \
        --selection-seed 1009 \
        --safety-margin 0.1 \
        --source-revision "$SOURCE_REVISION" \
        --output "$PREFLIGHT"
fi

python - "$PREFLIGHT" "$SCENARIO_CONFIG" "$TRACE" \
    "$IMPORTANCE_OUTPUT" "$SOURCE_REVISION" <<'PY'
import hashlib
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())


def digest(raw_path):
    return hashlib.sha256(pathlib.Path(raw_path).read_bytes()).hexdigest()

assert payload["source_revision"] == sys.argv[5]
assert payload["inputs"]["scenario_config"]["sha256"] == digest(sys.argv[2])
assert payload["inputs"]["system_trace"]["sha256"] == digest(sys.argv[3])
assert payload["inputs"]["modality_importance"]["sha256"] == digest(sys.argv[4])
methods = payload["methods"]
full = methods["mics_v2_impact"]
without = methods["mics_v2_impact_no_client_queue"]
assert full["minimum_selection_count"] > 0
assert full["jain_selection_fairness"] > (
    without["jain_selection_fairness"] + 0.15
)
assert payload["client_targets"]
PY

COMMON_ARGUMENTS=(
    --dataset-root "$DATASET_ROOT"
    --scenario-config "$SCENARIO_CONFIG"
    --system-trace "$TRACE"
    --output-root "$STUDY_ROOT"
    --study-id "$STUDY_ID"
    --protocol-manifest "$MANIFEST"
    --framework-version v2
    --rounds 150
    --clients-per-round 2
    --checkpoint-every 10
    --service-type pure_coverage
    --saturation-level 1
    --safety-margin 0.1
    --reliability-aware
    --base-channels 16
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
    --model-seed 101
    --selection-seed 1009
    --data-seed 2003
    --device cuda
)

run_variant() {
    local run_id=$1
    local run_dir=$2
    shift 2
    local resume_argument=()
    if [[ -d "$run_dir" ]]; then
        python - "$run_dir/run_metadata.json" "$SOURCE_REVISION" <<'PY'
import json
import pathlib
import sys

metadata = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert metadata["source_revision"] == sys.argv[2], (
    "Refusing to reuse a v2 pilot run under a different source revision."
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
    --importance-output "$IMPORTANCE_OUTPUT" --impact-weight 1
run_variant "$CORE_ID" "$CORE_DIR" --impact-weight 0
run_variant "$NO_CLIENT_ID" "$NO_CLIENT_DIR" \
    --importance-output "$IMPORTANCE_OUTPUT" --impact-weight 1 \
    --ablation no_client_queue

evaluate_final() {
    local run_dir=$1
    local output=$run_dir/evaluations/validation_after_round_0150.json
    if [[ -f "$output" ]]; then
        echo "Evaluation already completed: $output"
        return
    fi
    python scripts/evaluate_brats_checkpoint.py \
        --run-dir "$run_dir" \
        --dataset-root "$DATASET_ROOT" \
        --scenario-config "$SCENARIO_CONFIG" \
        --checkpoint "$run_dir/checkpoints/after_round_0150.pt" \
        --partition validation \
        --roi-size 96 96 96 \
        --overlap 0.5 \
        --device cuda \
        --output "$output"
}

evaluate_final "$FULL_DIR"
evaluate_final "$CORE_DIR"
evaluate_final "$NO_CLIENT_DIR"

FULL_EVALUATION=$FULL_DIR/evaluations/validation_after_round_0150.json
CORE_EVALUATION=$CORE_DIR/evaluations/validation_after_round_0150.json
NO_CLIENT_EVALUATION=$NO_CLIENT_DIR/evaluations/validation_after_round_0150.json

if [[ ! -f "$REPORT" ]]; then
    python scripts/compare_method_matrix.py \
        --method mics_v2_impact "$FULL_DIR" "$FULL_EVALUATION" \
        --method mics_v2_core "$CORE_DIR" "$CORE_EVALUATION" \
        --method mics_v2_impact_no_client_queue \
          "$NO_CLIENT_DIR" "$NO_CLIENT_EVALUATION" \
        --reference-label mics_v2_impact \
        --output "$REPORT"
fi

python - "$PREFLIGHT" "$REPORT" <<'PY'
import json
import pathlib
import sys

preflight = json.loads(pathlib.Path(sys.argv[1]).read_text())
report = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert report["purpose"] == "matched_completed_method_matrix_comparison"
assert report["interpretation_allowed"] is True
assert report["reference_label"] == "mics_v2_impact"
for label in report["method_order"]:
    method = report["methods"][label]
    selection = method["selection_configuration"]
    expected_queue = label != "mics_v2_impact_no_client_queue"
    assert selection["framework_version"] == "v2"
    assert selection["client_queue_enabled"] is expected_queue
    assert method["selection_and_training"]["selection_counts"] == (
        preflight["methods"][label]["selection_counts"]
    )
assert report["methods"]["mics_v2_impact"]["selection_configuration"][
    "client_targets"
] == report["methods"]["mics_v2_core"]["selection_configuration"][
    "client_targets"
]
PY

echo "Selector preflight: $PREFLIGHT"
echo "Pilot report: $REPORT"
echo "MICS-V2 CLIENT-QUEUE SINGLE-SEED PILOT PASSED"
