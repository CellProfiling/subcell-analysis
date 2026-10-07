#!/bin/bash
#SBATCH --job-name=subcell_inf
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --array=0-59%4
#SBATCH --output=slurm_out/subcell_%A_%a.out
#SBATCH --error=slurm_out/subcell_%A_%a.err
#
# Array job: 15 plates x 2 SubCell variants x 2 mask modes = 60 tasks.
#   task_id // 30 -> mask mode   (0=unmasked, 1=masked)
#   (task_id % 30) // 15 -> config index (0=mae, 1=vit)
#   task_id % 15  -> plate index into plate_list
#
# Outputs land in {model_name}/ for unmasked and {model_name}_masked/ for masked,
# so both halves can run concurrently without collision.


set -euo pipefail

set -a
source ../.env
set +a

set +u
module load cuda/12.8.0
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_INFERENCE}"
set -u

CONFIGS=(../configs/subcell_mae.yaml ../configs/subcell_vit.yaml)
MASK_IDX=$(( SLURM_ARRAY_TASK_ID / 30 ))
CONFIG_IDX=$(( (SLURM_ARRAY_TASK_ID % 30) / 15 ))
PLATE_IDX=$(( SLURM_ARRAY_TASK_ID % 15 ))
CONFIG=${CONFIGS[$CONFIG_IDX]}
MASK_FLAG=$([ "$MASK_IDX" -eq 1 ] && echo "--apply-mask" || echo "")

PLATE=$(python -c "
import yaml
with open('../configs/plates.yaml') as f:
    print(yaml.safe_load(f)['plate_list'][${PLATE_IDX}])
")

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CONFIG} plate=${PLATE} mask=${MASK_FLAG:-off}"

srun python ../scripts/inference_subcell.py \
    --config "${CONFIG}" \
    --plate "${PLATE}" \
    --cell-batch-size 32 \
    ${MASK_FLAG}
