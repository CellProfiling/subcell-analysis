#!/bin/bash
#SBATCH --job-name=postproc
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --output=slurm_out/postproc_%j.out
#SBATCH --error=slurm_out/postproc_%j.err
#
# Run the full postprocessing grid: 2 feat_sel x 18 norm combos per
# (input CSV, cell_type). At ~1-2s/combo this is well under an hour
# for the small embeddings; CellProfiler (5792D) will dominate.
#
# Resumable — skips existing parquets.

set -euo pipefail

set -a
source ../.env
set +a

set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_ANALYSIS}"
set -u

# No model list = every CSV in EMBED_WELL_ROOT (all models, both aggs)
python ../scripts/postprocess.py
