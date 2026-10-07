#!/bin/bash
#SBATCH --job-name=merge_map
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH -t 00:05:00
#SBATCH --output=slurm_out/merge_map_%j.out
#SBATCH --error=slurm_out/merge_map_%j.err
#
# Merge per-model results_summary_*.csv into results_summary.csv.
# Run after all 08_compute_map array tasks finish.
#
# Usage: sbatch --dependency=afterok:<array_job_id> 08b_merge_map_summaries.sh

set -euo pipefail

set -a
source ../.env
set +a

set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_ANALYSIS}"
set -u

python -c "
import pandas as pd
from glob import glob
import os

results_root = os.environ['RESULTS_ROOT']
parts = sorted(glob(os.path.join(results_root, 'results_summary_*.csv')))
if not parts:
    print('No per-model summaries found')
    exit(1)

dfs = [pd.read_csv(p) for p in parts]
for p in parts:
    print(f'  {os.path.basename(p)}: {len(pd.read_csv(p))} rows')

merged = pd.concat(dfs, ignore_index=True)
merged = merged.drop_duplicates(
    subset=['model', 'cell_type', 'agg', 'fs', 'norm1', 'norm2', 'task'],
    keep='last',
)
out = os.path.join(results_root, 'results_summary.csv')
merged.to_csv(out, index=False)
print(f'Merged {len(merged)} rows -> {out}')
"
