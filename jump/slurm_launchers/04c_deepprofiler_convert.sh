#!/bin/bash
#SBATCH --job-name=dp_convert
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --output=slurm_out/dp_convert_%j.out
#SBATCH --error=slurm_out/dp_convert_%j.err
#
# Convert DeepProfiler .npz features to per-well .pth files
# for consumption by aggregate_cells.py.
# Usage: sbatch 04c_deepprofiler_convert.sh

set -euo pipefail

set -a
source ../.env
set +a

set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_ANALYSIS}"
set -u

python ../scripts/convert_deepprofiler_to_pth.py \
    --dp-features "${DP_ROOT}/outputs/results/features/"
