#!/bin/bash
#SBATCH --job-name=mics-refbase
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=mics-baselines-%j.out
#SBATCH --error=mics-baselines-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
DATASET_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}
BUNDLE_ROOT=${MICS_SPLIT_BUNDLE_ROOT:-$HOME/pmcs/split/generated/data_seed_20260905}
SCENARIO_CONFIG=${MICS_SCENARIO_CONFIG:-$BUNDLE_ROOT/balanced/materialized_config.json}
RESULTS_ROOT=${MICS_RESULTS_ROOT:-$PROJECT_ROOT/results}
POWER_RUN_ID=power-choice-balanced-smoke-${SLURM_JOB_ID}
DIVFL_RUN_ID=divfl-balanced-smoke-${SLURM_JOB_ID}
POWER_RUN_DIR=$RESULTS_ROOT/$POWER_RUN_ID
DIVFL_RUN_DIR=$RESULTS_ROOT/$DIVFL_RUN_ID

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$DATASET_ROOT"
test -f "$SCENARIO_CONFIG"
test ! -e "$POWER_RUN_DIR"
test ! -e "$DIVFL_RUN_DIR"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Source revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/validate_install.py

COMMON_ARGUMENTS=(
    --dataset-root "$DATASET_ROOT"
    --scenario-config "$SCENARIO_CONFIG"
    --output-root "$RESULTS_ROOT"
    --rounds 1
    --clients-per-round 2
    --checkpoint-every 1
    --probe-max-batches 1
    --base-channels 8
    --num-levels 4
    --fusion mean_var
    --modality-dropout 0.2
    --patch-size 96 96 96
    --batch-size 1
    --num-workers 0
    --local-epochs 1
    --max-batches-per-epoch 1
    --model-seed 101
    --selection-seed 202
    --data-seed 303
    --device cuda
)

python scripts/run_reference_baseline.py \
    --method power_of_choice \
    --run-id "$POWER_RUN_ID" \
    --candidate-size 4 \
    "${COMMON_ARGUMENTS[@]}"

python scripts/run_reference_baseline.py \
    --method divfl \
    --run-id "$DIVFL_RUN_ID" \
    "${COMMON_ARGUMENTS[@]}"

python - "$POWER_RUN_DIR" "$DIVFL_RUN_DIR" <<'PY'
import json
import pathlib
import sys

expected = ("local_loss", "gradient_vector")
for raw_path, expected_kind in zip(sys.argv[1:], expected):
    run_dir = pathlib.Path(raw_path)
    status = json.loads((run_dir / "status.json").read_text())
    records = [
        json.loads(line)
        for line in (run_dir / "round_metrics.jsonl").read_text().splitlines()
    ]
    assert status["state"] == "completed"
    assert len(records) == 1
    record = records[0]
    assert record["selection_probe"]["kind"] == expected_kind
    assert record["selection_probe"]["partition"] == "train"
    assert record["selection_probe"]["model_updated"] is False
    assert record["communication"]["selection_probe_download_bytes"] > 0
    assert record["communication"]["selection_probe_upload_bytes"] > 0
    assert (run_dir / "checkpoints" / "last.pt").is_file()
PY

echo "Power-of-Choice run directory: $POWER_RUN_DIR"
echo "DivFL-style run directory: $DIVFL_RUN_DIR"
echo "MICS PROBE-BASED REFERENCE BASELINES GPU JOB PASSED"
