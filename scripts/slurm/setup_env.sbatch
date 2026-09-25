#!/bin/bash
# Build the project virtualenv on the cluster (submitted, never run interactively).
#SBATCH --job-name=eg606-venv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/venv-%j.out
#SBATCH --error=logs/venv-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load anaconda3-2024
python -m venv .venv
.venv/bin/pip install -q -U pip
# L40S + driver 570 supports CUDA 12.8 wheels
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -q -e ".[prep,audio,train,dev]"
.venv/bin/python - <<'PY'
import torch, mne, librosa
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("mne", mne.__version__, "| librosa", librosa.__version__)
PY
echo VENV_OK
