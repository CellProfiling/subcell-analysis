#!/bin/bash
#SBATCH --job-name=attn_maps
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --output=slurm_out/attn_maps_%j.out
#SBATCH --error=slurm_out/attn_maps_%j.err
#
# Generate SubCell attention-map figures for JUMP.
# 2 cell types x 2 models = 4 runs, each emits 4 figures (Mito/AGP/RNA + Avg)
# -> 16 figures total written to figures/ as both PNG and PDF @ dpi=300.
#
# Crops were regenerated after the original figures, so random selection no
# longer picks the same crop files. CROPS_CSV lists the exact crops shown in
# those figures (recovered by pixel-matching the old PDFs, seed 0).
#
# Submit from slurm_launchers/ (paths below are relative to it):
#   cd $PROJECT_ROOT/slurm_launchers && sbatch 10_attention_maps_subcell.sh [seed]

set -euo pipefail

set -a
source ../.env
set +a

set +u
module load cuda/12.8.0
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_INFERENCE}"
set -u

SEED=${1:-0}
OUTDIR="${PROJECT_ROOT}/figures/attention/"
CROPS_CSV="${PROJECT_ROOT}/configs/attention_crops_seed0.csv"

for CELL_TYPE in A549 U2OS; do
    for MODEL in mae vit; do
        echo "[$(date)] cell_type=${CELL_TYPE} model=${MODEL} seed=${SEED}"
        python ../scripts/jump_attention_map_subcell.py \
            --cell-type "${CELL_TYPE}" \
            --model "${MODEL}" \
            --protein all \
            --seed "${SEED}" \
            --n-poscon 8 \
            --crops-csv "${CROPS_CSV}" \
            --output-dir "${OUTDIR}"
    done
done

echo "[$(date)] Done. Figures written to ${OUTDIR}"
