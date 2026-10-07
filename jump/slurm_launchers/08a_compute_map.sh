#!/bin/bash
#SBATCH --job-name=compute_map
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm_out/map_%A_%a.out
#SBATCH --error=slurm_out/map_%A_%a.err
#
# Compute copairs mAP (replicate + MoA) for every processed parquet.
# One array task per model, each writes to its own summary CSV.
# After all tasks finish, merge with 08b_merge_map_summaries.sh.
#
# Usage: sbatch 08_compute_map.sh

set -euo pipefail

set -a
source ../.env
set +a

set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_ANALYSIS}"
set -u

# 12 models = 12 tasks (array 0-11)
MODELS=(cellprofiler deepprofiler dino dino_masked subcell_mae subcell_mae_masked subcell_vit subcell_vit_masked
        cell_dino cell_dino_avgpool cell_dino_masked cell_dino_avgpool_masked)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] compute_map model=${MODEL}"
python ../scripts/compute_map.py \
    --model "${MODEL}" \
    --output-suffix "${MODEL}" \
    --null-size 10000
