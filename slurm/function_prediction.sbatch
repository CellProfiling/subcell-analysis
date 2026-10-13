#!/bin/bash -l
#SBATCH --job-name=function_prediction
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH -G 1
# several unit processes share the GPU, which needs shared compute mode (drop the
# next line on sites without gpu_cmode)
#SBATCH --gpu_cmode=shared
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --requeue

# One GPU running `python -m subcell.classification.function_prediction run`.
# Every runner shares one unit queue through claim files, so any number of these
# can run at once, e.g. a 16-task array (set the log path with -o; the directory
# must exist):
#   mkdir -p "$SUBCELL_ROOT/logs"
#   sbatch -o "$SUBCELL_ROOT/logs/%x_%A_%a.out" --array=0-15 slurm/function_prediction.sbatch
#
# Environment: SUBCELL_PYTHON is the interpreter of the analysis environment
# (default: `python` on PATH); SUBCELL_ROOT and the other roots are those of
# subcell/config.py. Submit from the repository root.
#
# A unit peaks near 6 GB of GPU memory, so a card under 20 GB runs one unit at a
# time and a larger one two.
#
# A pre-empted task is requeued; finished folds are skipped, and its claims expire
# after REMOTE_CLAIM_TTL (see subcell/classification/function_prediction.py).

set -euo pipefail

# Optional, if the interpreter needs them:
# module load cuda python
# export LD_LIBRARY_PATH=<libraries the interpreter needs>:$LD_LIBRARY_PATH

cd "$SLURM_SUBMIT_DIR"

echo "start=$(date +%s) host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
     "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$MEM_MB" -lt 20000 ]; then
    SLOTS=1; CAP_GB=$(( (MEM_MB - 1500) / 1024 ))
elif [ "$MEM_MB" -lt 30000 ]; then
    SLOTS=2; CAP_GB=10
else
    SLOTS=2; CAP_GB=12
fi
GPUS=$(for _ in $(seq "$SLOTS"); do printf '%s ' "$CUDA_VISIBLE_DEVICES"; done)
echo "slots=$SLOTS cap=${CAP_GB}GB"
# $GPUS unquoted on purpose: one word per slot
${SUBCELL_PYTHON:-python} -u -m subcell.classification.function_prediction run \
    --gpus $GPUS --gpu_mem_gb "$CAP_GB" --n_jobs 3
