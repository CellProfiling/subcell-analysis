#!/bin/bash
#SBATCH --job-name=dino_inf
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH --array=0-29%4
#SBATCH --output=slurm_out/dino_%A_%a.out
#SBATCH --error=slurm_out/dino_%A_%a.err
#
# Array job: 15 plates x 2 mask modes = 30 tasks.
#   task_id // 15 -> mask mode (0=unmasked, 1=masked)
#   task_id % 15  -> plate index
# DINO is ~10x cheaper than SubCell (1 pass vs 3, 128px vs 478px)
# so shorter wall time and larger batch.
#
# Outputs: dino/ (unmasked), dino_masked/ (masked).

set -euo pipefail

set -a
source ../.env
set +a

set +u
module load cuda/12.8.0
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_INFERENCE}"
set -u

MASK_IDX=$(( SLURM_ARRAY_TASK_ID / 15 ))
PLATE_IDX=$(( SLURM_ARRAY_TASK_ID % 15 ))
MASK_FLAG=$([ "$MASK_IDX" -eq 1 ] && echo "--apply-mask" || echo "")

PLATE=$(python -c "
import yaml
with open('../configs/plates.yaml') as f:
    print(yaml.safe_load(f)['plate_list'][${PLATE_IDX}])
")

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} plate=${PLATE} mask=${MASK_FLAG:-off}"

srun python ../scripts/inference_dino.py \
    --config ../configs/dino.yaml \
    --plate "${PLATE}" \
    --cell-batch-size 128 \
    ${MASK_FLAG}
