#!/bin/bash
#SBATCH --job-name=attn_dino
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH --array=0-1
#SBATCH --output=slurm_out/attn_dino_%A_%a.out
#SBATCH --error=slurm_out/attn_dino_%A_%a.err
#
# Generate DINO4Cells attention-map figures for JUMP.
# Job array: task 0 = native resolution, task 1 = upscaled 3.74x
# (matching SubCell's rescale_ratio so attention grids are comparable).
#
# Each task produces 2 figures (A549 + U2OS) as PNG + PDF @ dpi=300.
#
# Pass the same seed you used for 08_attention_maps.sh to visualize the
# same cells — pick_compounds/find_crop_for_compound are byte-identical
# and consume np.random.default_rng(seed) in the same order.
#
# Crops were regenerated after the original figures, so random selection no
# longer picks the same crop files. CROPS_CSV lists the exact crops shown in
# those figures (recovered by pixel-matching the old PDFs, seed 0).
#
# Submit from slurm_launchers/ (paths below are relative to it):
#   cd $PROJECT_ROOT/slurm_launchers && sbatch 10_attention_maps_dino.sh [seed]

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

UPSCALE_FACTORS=(1.0 3.74)
UPSCALE=${UPSCALE_FACTORS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] array_task=${SLURM_ARRAY_TASK_ID} upscale_factor=${UPSCALE}"

for CELL_TYPE in A549 U2OS; do
    echo "[$(date)] cell_type=${CELL_TYPE} seed=${SEED} upscale=${UPSCALE}"
    python ../scripts/jump_attention_map_dino.py \
        --cell-type "${CELL_TYPE}" \
        --seed "${SEED}" \
        --n-poscon 8 \
        --upscale-factor "${UPSCALE}" \
        --crops-csv "${CROPS_CSV}" \
        --output-dir "${OUTDIR}"
done

echo "[$(date)] Done. Figures written to ${OUTDIR}"
