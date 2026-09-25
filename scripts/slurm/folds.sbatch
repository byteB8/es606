#!/bin/bash
# Array job: one leave-one-subject-out fold per array task.
#   sbatch --array=0-19 scripts/slurm/folds.sbatch "python -m eg606.train fold=\$SLURM_ARRAY_TASK_ID"
#SBATCH --job-name=eg606-folds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.out

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else module load anaconda3-2024; PY=python; fi
export EG606_DATA="${EG606_DATA:-$HOME/egdta}"
export PYTHONPATH=src
echo "fold=$SLURM_ARRAY_TASK_ID host=$(hostname)"
srun $PY "$@"
