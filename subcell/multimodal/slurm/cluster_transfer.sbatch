#!/bin/bash -l
#SBATCH --job-name=cluster_transfer
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# `python -m subcell.multimodal.cluster_transfer` and `cluster_percentile`, then their
# figures (Fig. 6a, Extended Data Fig. 7a). On 4 CPUs the 10,363 proteins of the U2OS run take about 4 minutes;
# the percentile ranks peak at 3.5 GB.
#   sbatch subcell/multimodal/slurm/cluster_transfer.sbatch [--run-dir DIR]
# Arguments are passed to the analysis.
#
# Environment: SUBCELL_PYTHON is the interpreter of the analysis environment
# (default: `python` on PATH); SUBCELL_ROOT and the other roots are those of
# subcell/config.py. Submit from the repository root.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

${SUBCELL_PYTHON:-python} -u -m subcell.multimodal.cluster_transfer "$@"
${SUBCELL_PYTHON:-python} -u -m subcell.multimodal.cluster_percentile "$@"
${SUBCELL_PYTHON:-python} -u -m figures.fig6a_cluster_transfer
${SUBCELL_PYTHON:-python} -u -m figures.ed7a_cluster_percentile
