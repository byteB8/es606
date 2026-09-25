#!/usr/bin/env bash
# Corrected-timing transfer experiments on the Singularity cluster.
#
# MUSIN-G needs a +62 ms correction (its event markers lag the sound); Bach and SparrKULee need
# none. Every earlier transfer result on MUSIN-G was measured against the uncorrected timing, so
# the speech-pretrained model was fighting a phase mismatch.
#
#   bash scripts/submit_shift.sh        # run ON the cluster, from ~/codes/eg606
set -euo pipefail
CKPT="$HOME/egdta/models/speech_v2_best.pt"
[ -f "$CKPT" ] || { echo "missing speech checkpoint: $CKPT" >&2; exit 1; }
mkdir -p logs results

sub () {  # sub <job-name> <args...>
  local name="$1"; shift
  sbatch --job-name="$name" scripts/slurm/train.sbatch "$*" | tail -1
}

# --- zero-shot evaluations (minutes each): does the speech model work once timing is fixed?
sub zs-m0   -m eg606.train.cv --zero-shot "$CKPT" --dataset music --shift-ms 0  --tag zs_music_shift0
sub zs-m62  -m eg606.train.cv --zero-shot "$CKPT" --dataset music --shift-ms 62 --tag zs_music_shift62
sub zs-bach -m eg606.train.cv --zero-shot "$CKPT" --dataset bach  --windows 5,10 --tag zs_bach

# --- the experiment: fine-tuning with corrected timing, against a scratch control
sub m-ft62  -m eg606.train.cv --dataset music --shift-ms 62 --cv 5 --init "$CKPT" --tag music_ft_shift62
sub m-sc62  -m eg606.train.cv --dataset music --shift-ms 62 --cv 5 --tag music_v2_shift62

# --- the same question on a dataset that never needed correction
sub b-ft     -m eg606.train.cv --dataset bach --cv 5 --windows 5,10 --init "$CKPT" --tag bach_ft
sub b-sc     -m eg606.train.cv --dataset bach --cv 5 --windows 5,10 --tag bach_scratch

squeue -u "$(whoami)"
