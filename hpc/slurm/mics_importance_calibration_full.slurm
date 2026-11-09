#!/bin/bash
#SBATCH --job-name=mics-calibrate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=mics-calibrate-%j.out
#SBATCH --error=mics-calibrate-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
DATASET_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}
BUNDLE_ROOT=${MICS_SPLIT_BUNDLE_ROOT:-$HOME/pmcs/split/generated/data_seed_20260905}
SOURCE_SCENARIO=${MICS_BALANCED_SCENARIO:-$BUNDLE_ROOT/balanced/materialized_config.json}
SCIENTIFIC_BUNDLE=${MICS_SCIENTIFIC_BUNDLE:-$BUNDLE_ROOT/balanced_calibration}
SCENARIO_CONFIG=$SCIENTIFIC_BUNDLE/materialized_config.json
MANIFEST=${MICS_STUDY_MANIFEST:-$PROJECT_ROOT/configs/mics_ablation_study_v1.json}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$DATASET_ROOT"
test -f "$SOURCE_SCENARIO"
test -f "$MANIFEST"
git diff --quiet
git diff --cached --quiet

eval "$(python scripts/resolve_ablation_study.py \
    --manifest "$MANIFEST" --calibration --format shell)"

STUDY_ROOT=$PROJECT_ROOT/results/studies/$STUDY_ID
TRACE=$STUDY_ROOT/calibration/pilot_trace.json
RUN_ID=importance_pilot
RUN_DIR=$STUDY_ROOT/$RUN_ID
IMPORTANCE_OUTPUT=$STUDY_ROOT/calibration/modality_importance.json

mkdir -p "$STUDY_ROOT/calibration"
if [[ ! -f "$SCENARIO_CONFIG" ]]; then
    python scripts/materialize_calibration_brats_scenario.py \
        --dataset-root "$DATASET_ROOT" \
        --source-config "$SOURCE_SCENARIO" \
        --output-directory "$SCIENTIFIC_BUNDLE" \
        --split-seed "$CALIBRATION_SPLIT_SEED"
fi

python scripts/validate_brats_scenarios.py \
    --dataset-root "$DATASET_ROOT" \
    --scenario-config "$SCENARIO_CONFIG"

if [[ ! -f "$TRACE" ]]; then
    python scripts/generate_system_trace.py \
        --dataset-root "$DATASET_ROOT" \
        --scenario-config "$SCENARIO_CONFIG" \
        --output "$TRACE" \
        --trace-id "$STUDY_ID-importance-pilot" \
        --rounds "$ROUNDS" \
        --seed "$TRACE_SEED" \
        --minimum-available-clients "$CLIENTS_PER_ROUND" \
        --availability-probability "$CALIBRATION_AVAILABILITY_PROBABILITY" \
        --completion-probability "$CALIBRATION_COMPLETION_PROBABILITY"
fi

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Source revision: $(git rev-parse HEAD)"
echo "Protocol: $PROTOCOL_PATH"
echo "Protocol SHA-256: $PROTOCOL_SHA256"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/validate_install.py

RESUME_ARGUMENT=()
if [[ -d "$RUN_DIR" ]]; then
    python - "$RUN_DIR/run_metadata.json" "$(git rev-parse HEAD)" <<'PY'
import json
import pathlib
import sys

metadata = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert metadata["source_revision"] == sys.argv[2], (
    "Refusing to resume calibration pilot under a different source revision."
)
assert metadata["source_dirty"] is False
PY
    if grep -q '"state": "completed"' "$RUN_DIR/status.json"; then
        echo "Calibration pilot training already completed: $RUN_DIR"
    else
        RESUME_ARGUMENT=(--resume)
    fi
fi

if [[ ! -f "$RUN_DIR/status.json" ]] || \
   ! grep -q '"state": "completed"' "$RUN_DIR/status.json"; then
    read -r PATCH_D PATCH_H PATCH_W <<< "$PATCH_SIZE"
    python scripts/run_uniform_baseline.py \
        --dataset-root "$DATASET_ROOT" \
        --scenario-config "$SCENARIO_CONFIG" \
        --system-trace "$TRACE" \
        --output-root "$STUDY_ROOT" \
        --run-id "$RUN_ID" \
        --study-id "$STUDY_ID" \
        --protocol-manifest "$PROTOCOL_PATH" \
        --rounds "$ROUNDS" \
        --clients-per-round "$CLIENTS_PER_ROUND" \
        --checkpoint-every "$CHECKPOINT_EVERY" \
        --base-channels "$BASE_CHANNELS" \
        --num-levels "$NUM_LEVELS" \
        --fusion "$FUSION" \
        --modality-dropout "$MODALITY_DROPOUT" \
        --patch-size "$PATCH_D" "$PATCH_H" "$PATCH_W" \
        --foreground-probability "$FOREGROUND_PROBABILITY" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --local-epochs "$LOCAL_EPOCHS" \
        --learning-rate "$LEARNING_RATE" \
        --weight-decay "$WEIGHT_DECAY" \
        --gradient-clip-norm "$GRADIENT_CLIP_NORM" \
        --amp-initial-scale "$AMP_INITIAL_SCALE" \
        --amp-max-retries "$AMP_MAX_RETRIES" \
        --model-seed "$MODEL_SEED" \
        --selection-seed "$SELECTION_SEED" \
        --data-seed "$DATA_SEED" \
        --device cuda \
        "${RESUME_ARGUMENT[@]}"
fi

if [[ ! -f "$IMPORTANCE_OUTPUT" ]]; then
    read -r ROI_D ROI_H ROI_W <<< "$ROI_SIZE"
    read -r WEIGHT_WT WEIGHT_TC WEIGHT_ET <<< "$REGION_WEIGHTS"
    python scripts/estimate_modality_importance.py \
        --run-dir "$RUN_DIR" \
        --dataset-root "$DATASET_ROOT" \
        --scenario-config "$SCENARIO_CONFIG" \
        --partition calibration \
        --roi-size "$ROI_D" "$ROI_H" "$ROI_W" \
        --overlap "$OVERLAP" \
        --region-weights "$WEIGHT_WT" "$WEIGHT_TC" "$WEIGHT_ET" \
        --device cuda \
        --output "$IMPORTANCE_OUTPUT"
fi

python - "$IMPORTANCE_OUTPUT" "$SCENARIO_CONFIG" \
    "$PROTOCOL_SHA256" "$STUDY_ID" <<'PY'
import hashlib
import json
import pathlib
import sys

importance_path = pathlib.Path(sys.argv[1])
scenario_path = pathlib.Path(sys.argv[2])
record = json.loads(importance_path.read_text())
importance = record["importance"]
scenario_sha = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
assert record["purpose"] == "selector_calibration_modality_importance"
assert importance["partition"] == "calibration"
assert importance["complete_calibration_probe"] is True
assert importance["test_data_accessed"] is False
assert record["scenario"]["sha256"] == scenario_sha
assert record["study"]["id"] == sys.argv[4]
assert record["study"]["protocol_sha256"] == sys.argv[3]
PY

echo "Scientific scenario: $SCENARIO_CONFIG"
echo "Calibration pilot: $RUN_DIR"
echo "Importance output: $IMPORTANCE_OUTPUT"
echo "MICS COMPLETE CALIBRATION-IMPORTANCE JOB PASSED"
