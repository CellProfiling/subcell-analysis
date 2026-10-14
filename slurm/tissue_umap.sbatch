#!/bin/bash -l
#SBATCH --job-name=tissue_umap
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=250GB

# UMAP of the uncorrected and each harmonized tissue feature matrix, on the top 256
# principal components (GPU randomized SVD) of each matrix, for
# figures/ed2i_tissue_umaps.py. Run after slurm/tissue_harmonization.sbatch.
#
#   sbatch -o "$SUBCELL_ROOT/logs/%x_%j.out" slurm/tissue_umap.sbatch
#
# SUBCELL_PYTHON selects the interpreter (default: python on PATH). A login shell
# is requested so that `module` is defined; load the modules your Python
# environment needs (scipy needs an OpenBLAS library) before the python call.
# The python call is not wrapped in srun: this is a single-task job.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date)"
# --fresh refits every embedding, replacing any cached at a different PCA dimension
"${SUBCELL_PYTHON:-python}" -m subcell.tissue.harmonization umap --fresh --pca-dim 256
echo "end=$(date)"
