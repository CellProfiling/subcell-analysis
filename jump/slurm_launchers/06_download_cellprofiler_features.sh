#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH --output=slurm_out/cell_profiler_%j.out
#SBATCH --error=slurm_out/cell_profiler_%j.err
#
# Fetch CellProfiler well-level profiles from S3 (CP_S3_BUCKET/CP_S3_PREFIX).
# Doesn't depend on inference — can run anytime.

set -euo pipefail

set -a
source ../.env
set +a

set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_ANALYSIS}"
set -u

echo "[$(date)] CellProfiler (S3)"
python ../scripts/download_cellprofiler.py
