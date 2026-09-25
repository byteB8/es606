#!/bin/bash
# Template Slurm job for the Singularity cluster (singularity.iitgn.ac.in, port 2022).
# Cluster policy: jobs ONLY through the scheduler (no interactive runs on any node),
# max 2 GPUs + 32 CPU cores, 5-day wall time per user.
#
#   sbatch scripts/slurm/train.sbatch "python -m eg606.<module> <args>"
#
#SBATCH --job-name=eg606
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

# Environment: prefer the project venv; fall back to the cluster's anaconda module.
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  module load anaconda3-2024
  PY=python
fi

export EG606_DATA="${EG606_DATA:-$HOME/egdta}"
export PYTHONPATH=src

echo "host=$(hostname) job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-none} data=$EG606_DATA"

# Sample GPU usage into its own file: the login node has no GPUs and the cluster forbids
# interactive jobs, so `tail -f logs/gpu-<jobid>.log` is how to watch utilisation.
( nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader --loop=30 > "logs/gpu-$SLURM_JOB_ID.log" 2>&1 ) &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
srun $PY ${@:-"-c" "print('no command given')"}
