#!/bin/bash -l
#SBATCH --job-name=tissue_harmonization
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB

# Harmony batch correction of the HPA tissue features over the candidate batch
# keys (Extended Data Fig. 2i, Supplementary Note 2): stages validate + harm, or
# all (harm then umap). Needs a GPU.
#
#   sbatch -o "$SUBCELL_ROOT/logs/%x_%j.out" slurm/tissue_harmonization.sbatch [stage]
#
# SUBCELL_PYTHON selects the interpreter (default: python on PATH). Add the
# `module load` lines your Python environment needs before the python call.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date)"
"${SUBCELL_PYTHON:-python}" -m subcell.tissue.harmonization "${1:-all}"
echo "end=$(date)"
