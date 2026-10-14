#!/bin/bash -l
#SBATCH --job-name=paralog_ranks
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# `python -m subcell.multimodal.paralog_ranks`, then its figures (Fig. 6e). On 4 CPUs
# the 10,363 proteins of the U2OS run take under a minute and about 3 GB.
#   sbatch subcell/multimodal/slurm/paralog_ranks.sbatch [--run-dir DIR]
# Arguments are passed to the analysis. The figures are drawn by figures.make_all from
# the results of the default run (subcell.multimodal.embeddings.METHOD_NAME); one whose
# inputs are missing is skipped.
#
# Environment: SUBCELL_PYTHON is the interpreter of the analysis environment
# (default: `python` on PATH); SUBCELL_ROOT and the other roots are those of
# subcell/config.py. Submit from the repository root.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

${SUBCELL_PYTHON:-python} -u -m subcell.multimodal.paralog_ranks "$@"
${SUBCELL_PYTHON:-python} -u -m figures.make_all fig6e
