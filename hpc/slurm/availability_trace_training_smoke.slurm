#!/bin/bash
#SBATCH --job-name=mics-avail
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=mics-avail-%j.out
#SBATCH --error=mics-avail-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
DATASET_ROOT=${MICS_DATA_ROOT:-$HOME/BraTS2020_npy}
BUNDLE_ROOT=${MICS_SPLIT_BUNDLE_ROOT:-$HOME/pmcs/split/generated/data_seed_20260905}
SCENARIO_CONFIG=${MICS_SCENARIO_CONFIG:-$BUNDLE_ROOT/balanced/materialized_config.json}
IMPORTANCE_OUTPUT=${MICS_IMPORTANCE_OUTPUT:?Set MICS_IMPORTANCE_OUTPUT to the completed importance JSON}
TRACE=$PROJECT_ROOT/results/availability-trace-smoke-${SLURM_JOB_ID}.json
UNIFORM_ID=uniform-availability-smoke-${SLURM_JOB_ID}
MICS_ID=mics-availability-smoke-${SLURM_JOB_ID}
UNIFORM_DIR=$PROJECT_ROOT/results/$UNIFORM_ID
MICS_DIR=$PROJECT_ROOT/results/$MICS_ID

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$DATASET_ROOT"
test -f "$SCENARIO_CONFIG"
test -f "$IMPORTANCE_OUTPUT"
test ! -e "$TRACE"
test ! -e "$UNIFORM_DIR"
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
    --trace-id availability-heterogeneity-smoke-${SLURM_JOB_ID} \
    --rounds 4 \
    --seed 505 \
    --minimum-available-clients 2 \
    --availability-probability 0.65 \
    --completion-probability 1.0

python scripts/run_uniform_baseline.py \
    --dataset-root "$DATASET_ROOT" \
    --scenario-config "$SCENARIO_CONFIG" \
    --system-trace "$TRACE" \
    --output-root "$PROJECT_ROOT/results" \
    --run-id "$UNIFORM_ID" \
    --rounds 4 \
    --clients-per-round 2 \
    --checkpoint-every 2 \
    --base-channels 16 \
    --num-levels 4 \
    --fusion mean_var \
    --modality-dropout 0.2 \
    --patch-size 96 96 96 \
    --batch-size 1 \
    --num-workers 0 \
    --local-epochs 1 \
    --max-batches-per-epoch 1 \
    --device cuda

python scripts/run_mics_training.py \
    --dataset-root "$DATASET_ROOT" \
    --scenario-config "$SCENARIO_CONFIG" \
    --system-trace "$TRACE" \
    --importance-output "$IMPORTANCE_OUTPUT" \
    --output-root "$PROJECT_ROOT/results" \
    --run-id "$MICS_ID" \
    --rounds 4 \
    --clients-per-round 2 \
    --checkpoint-every 2 \
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
    --max-batches-per-epoch 1 \
    --device cuda

grep -q '"state": "completed"' "$UNIFORM_DIR/status.json"
grep -q '"state": "completed"' "$MICS_DIR/status.json"
test -f "$UNIFORM_DIR/checkpoints/after_round_0004.pt"
test -f "$MICS_DIR/checkpoints/after_round_0004.pt"
grep -q '"availability_conditioned_exact_service_mixture_lp"' \
    "$MICS_DIR/resolved_config.json"
grep -q '"available_clients":\["C1","C2","C4","C5","C6","C7","C8"\]' \
    "$MICS_DIR/round_metrics.jsonl"
echo "Availability trace: $TRACE"
echo "Uniform run: $UNIFORM_DIR"
echo "MICS run: $MICS_DIR"
echo "MICS MATCHED AVAILABILITY-TRACE TRAINING SMOKE JOB PASSED"
