#!/bin/bash
#SBATCH --job-name=aggregate
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH --output=slurm_out/aggregate_%A_%a.out
#SBATCH --error=slurm_out/aggregate_%A_%a.err
#SBATCH --array=0-21
#
# Aggregate cell-level embeddings to well-level for all 3 neural models,
# both mean and median. Run after launchers 01 and 02 finish.
set -euo pipefail
set -a
source ../.env
set +a
set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_ANALYSIS}"
set -u

# 11 models x 2 aggs = 22 tasks (array 0-21)
models=(deepprofiler subcell_mae subcell_mae_masked subcell_vit subcell_vit_masked dino dino_masked
        cell_dino cell_dino_avgpool cell_dino_masked cell_dino_avgpool_masked)
aggs=(mean median)

model=${models[$((SLURM_ARRAY_TASK_ID / ${#aggs[@]}))]}
agg=${aggs[$((SLURM_ARRAY_TASK_ID % ${#aggs[@]}))]}

echo "[$(date)] ${model} ${agg}"
python ../scripts/aggregate_cells.py --model "${model}" --agg "${agg}"