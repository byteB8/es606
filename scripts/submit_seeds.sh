#!/usr/bin/env bash
# Replicate every arm that carries a claim with two further seeds (seed 0 is already done).
#
# Submitted as a SINGLE throttled array: the cluster allows 2 GPUs per user, and "%2" makes Slurm
# enforce that regardless of how many tasks are queued. Do not submit these as separate jobs.
set -euo pipefail
CKPT="$HOME/egdta/models/speech_v2_best.pt"
[ -f "$CKPT" ] || { echo "missing checkpoint $CKPT" >&2; exit 1; }
mkdir -p logs results
jobs=logs/seed_jobs.txt
: > "$jobs"

for seed in 1 2; do
  for arm in "--cv 5:music_lcv" "--cv-by song --cv 6:music_songcv" "--cv-by both --cv 6:music_bothcv"; do
    opts="${arm%%:*}"; tag="${arm##*:}"
    echo "-m eg606.train.cv --dataset music --shift-ms 62 $opts --seed $seed --tag ${tag}_s$seed" >> "$jobs"
    echo "-m eg606.train.cv --dataset music --shift-ms 62 $opts --seed $seed --init $CKPT --tag ${tag}_ft_s$seed" >> "$jobs"
  done
  echo "-m eg606.train.cv --dataset bach --cv 5 --windows 5,10 --seed $seed --tag bach_lcv_s$seed" >> "$jobs"
  echo "-m eg606.train.cv --dataset bach --cv 5 --windows 5,10 --seed $seed --init $CKPT --tag bach_lcv_ft_s$seed" >> "$jobs"
done

n=$(wc -l < "$jobs")
echo "$n tasks queued, at most 2 running at a time:"
cat -n "$jobs" | sed 's/^/  /'
sbatch --array=0-$((n - 1))%2 scripts/slurm/array.sbatch "$jobs"
