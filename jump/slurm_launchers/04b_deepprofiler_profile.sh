#!/bin/bash
#SBATCH --job-name=dp_profile
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 3-00:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_out/dp_profile_%A_%a.out
#SBATCH --error=slurm_out/dp_profile_%A_%a.err
#
# DeepProfiler profile: GPU feature extraction with Cell Painting CNN.
# Index pre-split into 6 parts (index-000.csv through index-005.csv).
# Each array task processes one part on its own GPU.
#
# Usage: sbatch 04b_deepprofiler_profile.sh

set -euo pipefail

set -a
source ../.env
set +a

source "${DEEPPROF_ENV}/bin/activate"

DEEPPROFILER="${PROJECT_ROOT}/models/DeepProfiler/deepprofiler"
# Lets DeepProfiler import its top-level plugins/ package from the submodule
export PYTHONPATH="${PROJECT_ROOT}/models/DeepProfiler:${PYTHONPATH:-}"

python "${DEEPPROFILER}" \
    --root="${DP_ROOT}" \
    --config=profiling.json \
    --metadata=index.csv \
    --exp=results \
    profile --part="${SLURM_ARRAY_TASK_ID}"
