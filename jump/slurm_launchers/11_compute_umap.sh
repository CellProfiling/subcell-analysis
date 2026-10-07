#!/bin/bash
#SBATCH --job-name=umap
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -t 4:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm_out/umap_%A_%a.out
#SBATCH --error=slurm_out/umap_%A_%a.err
#
# Array job: UMAP_MODELS x 2 cell types. Each model's postprocessing config is
# read from configs/BEST_POSTPROC_CONFIGS.yaml (written by
# visualize_postprocessing.ipynb).
#
#   task_id // 2 -> model index into UMAP_MODELS
#   task_id % 2  -> cell type (0=A549, 1=U2OS)
#
# --array must cover n_models * 2 tasks; update it (or pass --array to sbatch)
# when UMAP_MODELS changes.
#
# Submit from slurm_launchers/ (paths below are relative to it):
#   cd $PROJECT_ROOT/slurm_launchers && sbatch 11_compute_umap.sh

set -euo pipefail

set -a
source ../.env
set +a

set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_ANALYSIS}"
set -u

# Models to compute UMAPs for
UMAP_MODELS=(cellprofiler dino subcell_mae deepprofiler cell_dino)
CELL_TYPES=(A549 U2OS)

MODEL=${UMAP_MODELS[$(( SLURM_ARRAY_TASK_ID / ${#CELL_TYPES[@]} ))]}
CELL_TYPE=${CELL_TYPES[$(( SLURM_ARRAY_TASK_ID % ${#CELL_TYPES[@]} ))]}

# Best config for this model, as a file stem, e.g. median__fs0__PCAcor__standardize
STEM=$(python -c "
import yaml
c = yaml.safe_load(open('../configs/BEST_POSTPROC_CONFIGS.yaml'))['${MODEL}']
print(f\"{c['agg']}__fs{c['fs']}__{c['norm1']}__{c['norm2']}\")
")

INPUT="${EMBED_PROCESSED_ROOT}/${MODEL}/${CELL_TYPE}/${STEM}.parquet"

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} model=${MODEL} cell_type=${CELL_TYPE} input=${INPUT}"
python ../scripts/compute_umap.py "${INPUT}" --n-neighbors 100 --min-dist 0.25
