#!/bin/bash
#SBATCH --job-name=mics-aggregate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=mics-aggregate-%j.out
#SBATCH --error=mics-aggregate-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
MANIFEST=${MICS_STUDY_MANIFEST:-$PROJECT_ROOT/configs/mics_ablation_study_v1.json}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"
git diff --quiet
git diff --cached --quiet

eval "$(python scripts/resolve_ablation_study.py \
    --manifest "$MANIFEST" --task-index 0 --format shell)"

STUDY_ROOT=$PROJECT_ROOT/results/studies/$STUDY_ID
OUTPUT=$STUDY_ROOT/ablation_multiseed_summary.json
CSV_OUTPUT=$STUDY_ROOT/ablation_multiseed_summary.csv

python scripts/aggregate_ablation_study.py \
    --manifest "$MANIFEST" \
    --seed-report s01 "$STUDY_ROOT/s01/ablation_matrix.json" \
    --seed-report s02 "$STUDY_ROOT/s02/ablation_matrix.json" \
    --seed-report s03 "$STUDY_ROOT/s03/ablation_matrix.json" \
    --seed-report s04 "$STUDY_ROOT/s04/ablation_matrix.json" \
    --seed-report s05 "$STUDY_ROOT/s05/ablation_matrix.json" \
    --output "$OUTPUT" \
    --csv-output "$CSV_OUTPUT"

echo "Aggregate JSON: $OUTPUT"
echo "Aggregate CSV: $CSV_OUTPUT"
echo "MICS FULL MULTI-SEED ABLATION STUDY PASSED"
