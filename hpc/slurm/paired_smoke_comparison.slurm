#!/bin/bash
#SBATCH --job-name=mics-compare
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=mics-compare-%j.out
#SBATCH --error=mics-compare-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
BASELINE_RUN_DIR=${MICS_BASELINE_RUN_DIR:?Set MICS_BASELINE_RUN_DIR}
CANDIDATE_RUN_DIR=${MICS_CANDIDATE_RUN_DIR:?Set MICS_CANDIDATE_RUN_DIR}
BASELINE_EVALUATION=${MICS_BASELINE_EVALUATION:?Set MICS_BASELINE_EVALUATION}
CANDIDATE_EVALUATION=${MICS_CANDIDATE_EVALUATION:?Set MICS_CANDIDATE_EVALUATION}
OUTPUT=$PROJECT_ROOT/results/paired-smoke-comparison-${SLURM_JOB_ID}.json

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

test -d "$BASELINE_RUN_DIR"
test -d "$CANDIDATE_RUN_DIR"
test -f "$BASELINE_EVALUATION"
test -f "$CANDIDATE_EVALUATION"
test ! -e "$OUTPUT"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Source revision: $(git rev-parse HEAD)"
python scripts/validate_install.py

python scripts/compare_tracked_runs.py \
    --baseline-run-dir "$BASELINE_RUN_DIR" \
    --candidate-run-dir "$CANDIDATE_RUN_DIR" \
    --baseline-evaluation "$BASELINE_EVALUATION" \
    --candidate-evaluation "$CANDIDATE_EVALUATION" \
    --baseline-label uniform_random \
    --candidate-label mics_impact \
    --output "$OUTPUT"

test -f "$OUTPUT"
grep -q '"purpose": "paired_integration_audit_not_model_quality"' "$OUTPUT"
grep -q '"interpretation_allowed": false' "$OUTPUT"
echo "Comparison output: $OUTPUT"
echo "MICS PAIRED SMOKE COMPARISON JOB PASSED"
