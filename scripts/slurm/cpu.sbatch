#!/bin/bash
# CPU-bound Slurm job (linear models are BLAS work). The cluster QOS requires at least one GPU per
# job (QOSMinGRES), so one is requested even though it goes unused.
#
#   sbatch scripts/slurm/cpu.sbatch "-m eg606.eval.gate_a --feature onset"
#
#SBATCH --job-name=eg606-cpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/cpu-%j.out
#SBATCH --error=logs/cpu-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else source /etc/profile.d/modules.sh; module load anaconda3-2024; PY=python; fi
export EG606_DATA="${EG606_DATA:-$HOME/egdta}"
export PYTHONPATH=src
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
echo "host=$(hostname) job=$SLURM_JOB_ID cpus=${SLURM_CPUS_PER_TASK:-?}"
srun $PY $@
