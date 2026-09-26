#!/bin/bash
# Throttled job array: runs a list of commands, at most N at a time.
#
# Cluster policy (and the HPC team's request) is a maximum of 2 GPUs per user, so these are
# submitted as ONE array with "%2", which Slurm enforces: never more than two tasks running.
#
#   sbatch --array=0-15%2 scripts/slurm/array.sbatch logs/jobs.txt
#
#SBATCH --job-name=eg606
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/a-%A_%a.out
#SBATCH --error=logs/a-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
jobs_file="${1:?usage: sbatch --array=0-N%2 array.sbatch <jobs-file>}"
cmd=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$jobs_file")
[ -n "$cmd" ] || { echo "no command on line $((SLURM_ARRAY_TASK_ID + 1))"; exit 0; }

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else source /etc/profile.d/modules.sh; module load anaconda3-2024; PY=python; fi
export EG606_DATA="${EG606_DATA:-$HOME/egdta}"
export PYTHONPATH=src
mkdir -p logs results

echo "host=$(hostname) array=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "cmd: $cmd"
( nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader --loop=30 \
    > "logs/gpu-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 ) &
trap 'kill %1 2>/dev/null' EXIT
srun $PY $cmd
