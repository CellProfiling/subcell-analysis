#!/bin/bash
#SBATCH --job-name=dp_prepare
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH --output=slurm_out/dp_prepare_%j.out
#SBATCH --error=slurm_out/dp_prepare_%j.err
#
# DeepProfiler prepare: illumination correction + compression (CPU-only).
# Usage: sbatch 04a_deepprofiler_prepare.sh

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
    --cores=16 \
    prepare
