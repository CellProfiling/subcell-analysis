#!/bin/bash
#SBATCH --job-name=celldino_inf
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -t 3-00:00:00
#SBATCH --array=0-29%4
#SBATCH --output=slurm_out/celldino_%A_%a.out
#SBATCH --error=slurm_out/celldino_%A_%a.err
#
# Array job: 15 plates x 2 mask modes = 30 tasks.
#   task_id // 15 -> mask mode (0=unmasked, 1=masked)
#   task_id % 15  -> plate index
# Cell-DINO is ViT-S/8 at 128px (257 tokens) — same cost ballpark as DINO4Cells.
#
# FEATURE_MODE=both writes cls and cls_avgpool from one forward pass:
# Outputs: cell_dino/, cell_dino_avgpool/, cell_dino_masked/, cell_dino_avgpool_masked/

set -euo pipefail

set -a
source ../.env
set +a

set +u
module load cuda/12.8.0
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_INFERENCE}"
set -u

FEATURE_MODE=both

MASK_IDX=$(( SLURM_ARRAY_TASK_ID / 15 ))
PLATE_IDX=$(( SLURM_ARRAY_TASK_ID % 15 ))
MASK_FLAG=$([ "$MASK_IDX" -eq 1 ] && echo "--apply-mask" || echo "")

PLATE=$(python -c "
import yaml
with open('../configs/plates.yaml') as f:
    print(yaml.safe_load(f)['plate_list'][${PLATE_IDX}])
")

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} plate=${PLATE} mask=${MASK_FLAG:-off} features=${FEATURE_MODE}"

srun python ../scripts/inference_cell_dino.py \
    --config ../configs/cell_dino.yaml \
    --plate "${PLATE}" \
    --feature-mode "${FEATURE_MODE}" \
    --cell-batch-size 256 \
    ${MASK_FLAG}
