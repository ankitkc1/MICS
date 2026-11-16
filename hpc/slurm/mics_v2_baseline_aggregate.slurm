#!/bin/bash
#SBATCH --job-name=mics-v2-summary
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=mics-v2-baselines-summary-%j.out
#SBATCH --error=mics-v2-baselines-summary-%j.err

set -euo pipefail

PROJECT_ROOT=${MICS_PROJECT_ROOT:-$HOME/MICS}
ENV_NAME=${MICS_CONDA_ENV:-mics}
MANIFEST=${MICS_BASELINE_MANIFEST:-$PROJECT_ROOT/configs/mics_v2_single_seed_baselines_v1.json}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"
python scripts/run_single_seed_comparison.py compare --manifest "$MANIFEST"
