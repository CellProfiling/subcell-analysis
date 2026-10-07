#!/bin/bash
#SBATCH --job-name=dl_subcell
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH --output=slurm_out/dl_subcell_%j.out
#SBATCH --error=slurm_out/dl_subcell_%j.err
#
# Downloads SubCell MAE and ViT model weights (ybg channel config)
# from s3://czi-subcell-public/ into models/SubCellPortable/models/

set -euo pipefail

set -a
source ../.env
set +a

set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_ANALYSIS}"
set -u

cd "${PROJECT_ROOT}/models/SubCellPortable"

python -c "
from model_loader import ensure_models_available
print('Downloading MAE weights...')
ensure_models_available('ybg', 'mae_contrast_supcon_model', embeddings_only=False, update_model=True)
print('Downloading ViT weights...')
ensure_models_available('ybg', 'vit_supcon_model', embeddings_only=False, update_model=True)
print('Done.')
"
