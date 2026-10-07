#!/bin/bash
#
# Submit DeepProfiler data preparation pipeline:
#   Step 1: setup    — create directory structure
#   Step 2: download — job array (15 plates) downloading images + nuclei from S3
#   Step 3: index    — build index.csv after all downloads complete
#
# Usage: bash 03_deepprofiler_data.sh

set -euo pipefail

# --- Step 1: Setup directory structure ---
SETUP_JOB=$(sbatch --parsable <<'SETUP'
#!/bin/bash
#SBATCH --job-name=dp_setup
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -t 00:05:00
#SBATCH --output=slurm_out/dp_setup_%j.out
#SBATCH --error=slurm_out/dp_setup_%j.err

set -euo pipefail

set -a
source ../.env
set +a

source "${DEEPPROF_ENV}/bin/activate"

DEEPPROFILER="${PROJECT_ROOT}/models/DeepProfiler/deepprofiler"
# Lets DeepProfiler import its top-level plugins/ package from the submodule
export PYTHONPATH="${PROJECT_ROOT}/models/DeepProfiler:${PYTHONPATH:-}"

python "${DEEPPROFILER}" --root="${DP_ROOT}" setup
SETUP
)

echo "Submitted setup job: ${SETUP_JOB}"

# --- Step 2: Download images + nuclei (job array, 1 per plate) ---
DOWNLOAD_JOB=$(sbatch --parsable --dependency=afterok:${SETUP_JOB} <<'DOWNLOAD'
#!/bin/bash
#SBATCH --job-name=dl_dp_inputs
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 1-00:00:00
#SBATCH --array=0-14
#SBATCH --output=slurm_out/dl_dp_%A_%a.out
#SBATCH --error=slurm_out/dl_dp_%A_%a.err

set -euo pipefail

set -a
source ../.env
set +a

set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_ANALYSIS}"
set -u

PLATE=$(python -c "
import yaml
with open('../configs/plates.yaml') as f:
    print(yaml.safe_load(f)['plate_list'][${SLURM_ARRAY_TASK_ID}])
")

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} plate=${PLATE}"

srun python ../scripts/download_deepprofiler_inputs.py --plate "${PLATE}" --dp-root "${DP_ROOT}"
DOWNLOAD
)

echo "Submitted download array job: ${DOWNLOAD_JOB} (depends on ${SETUP_JOB})"

# --- Step 3: Build index.csv after all downloads finish ---
INDEX_JOB=$(sbatch --parsable --dependency=afterok:${DOWNLOAD_JOB} <<'INDEX'
#!/bin/bash
#SBATCH --job-name=dp_index
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH --output=slurm_out/dp_index_%j.out
#SBATCH --error=slurm_out/dp_index_%j.err

set -euo pipefail

set -a
source ../.env
set +a

set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_ANALYSIS}"
set -u

srun python ../scripts/build_deepprofiler_index.py --dp-root "${DP_ROOT}"
INDEX
)

echo "Submitted index job: ${INDEX_JOB} (depends on ${DOWNLOAD_JOB})"
