#!/bin/bash -l
#SBATCH --job-name=compartment_enrichment
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# `python -m subcell.multimodal.compartment_enrichment`, then its figures (Fig. 6c,f),
# with the labels shipped in annotations/. On 4 CPUs the three label sets take about
# 8 minutes and 2 GB.
#   sbatch subcell/multimodal/slurm/compartment_enrichment.sbatch [--run-dir DIR]
# Arguments are passed to the analysis.
#
# Environment: SUBCELL_PYTHON is the interpreter of the analysis environment
# (default: `python` on PATH); SUBCELL_ROOT and the other roots are those of
# subcell/config.py. Submit from the repository root.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

${SUBCELL_PYTHON:-python} -u -m subcell.multimodal.compartment_enrichment "$@"
${SUBCELL_PYTHON:-python} -u -m figures.fig6c_compartment_enrichment
${SUBCELL_PYTHON:-python} -u -m figures.fig6f_process_enrichment
