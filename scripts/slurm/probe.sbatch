#!/bin/bash
# One-off probe: what can a compute node actually do?
#SBATCH --job-name=eg606-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/probe-%j.out
#SBATCH --error=logs/probe-%j.out
set -u
echo "host=$(hostname) date=$(date)"
echo "--- gpu"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head
echo "--- cuda modules"; source /etc/profile.d/modules.sh 2>/dev/null; module load anaconda3-2024 2>&1; python --version
echo "--- internet from compute node"
for u in https://pypi.org/simple/ https://huggingface.co https://download.pytorch.org; do
  printf "%-34s %s\n" "$u" "$(curl -s -o /dev/null -w '%{http_code}' --max-time 12 "$u" || echo FAIL)"
done
echo "--- filesystems"; df -h /iitgn "$HOME" 2>/dev/null | tail -3
echo "--- local scratch"; ls -d /tmp /scratch /local 2>/dev/null; df -h /tmp | tail -1
