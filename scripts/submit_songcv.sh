#!/usr/bin/env bash
# Generalisation to unheard music: leave-one-song-out, and the joint listener+song test.
#
# Listener-CV answers "new listener, songs seen in training". A reviewer will ask whether the model
# recognises familiar songs instead of tracking music, which is what these runs answer. MUSIN-G has
# 12 songs (6 folds of 2), Bach only 4 melodies (4 folds of 1, so its training set is thin).
set -euo pipefail
CKPT="$HOME/egdta/models/speech_v2_best.pt"
[ -f "$CKPT" ] || { echo "missing checkpoint $CKPT" >&2; exit 1; }
mkdir -p logs results

sub () { local name="$1"; shift; sbatch --job-name="$name" scripts/slurm/train.sbatch "$*" | tail -1; }

# --- new song, listeners seen
sub ms-sc  -m eg606.train.cv --dataset music --shift-ms 62 --cv-by song --cv 6 --tag music_songcv
sub ms-ft  -m eg606.train.cv --dataset music --shift-ms 62 --cv-by song --cv 6 --init "$CKPT" --tag music_songcv_ft
sub bs-sc  -m eg606.train.cv --dataset bach --cv-by song --cv 4 --val-groups 1 --windows 5,10 --tag bach_songcv
sub bs-ft  -m eg606.train.cv --dataset bach --cv-by song --cv 4 --val-groups 1 --windows 5,10 --init "$CKPT" --tag bach_songcv_ft

# --- hardest: new listener AND new song
sub mb-sc  -m eg606.train.cv --dataset music --shift-ms 62 --cv-by both --cv 6 --tag music_bothcv
sub mb-ft  -m eg606.train.cv --dataset music --shift-ms 62 --cv-by both --cv 6 --init "$CKPT" --tag music_bothcv_ft

squeue -u "$(whoami)" -o "%.8i %.12j %.2t %.10M %R"
