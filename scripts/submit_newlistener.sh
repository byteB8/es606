#!/usr/bin/env bash
# The "which song is a new listener hearing?" experiments, on the Singularity cluster.
#
#   A  decoder at 64 Hz (all music targets; new listener, and new listener + new song),
#      CNN and logistic-regression baselines, and the 128 Hz preprocessing (0.5-45 and 0.5-32 Hz)
#   B  decoder at 128 Hz (0.5-32, 0.5-45) and on the 32-45 Hz gamma envelope; starts after A
#
# Each is ONE array throttled to 2 tasks, and B waits for all of A, so no more than 2 GPUs are
# ever held (cluster policy). Every job must hold a GPU (QOS), even the CPU-bound linear ones.
# Run on the cluster login node from ~/codes/eg606:  bash scripts/submit_newlistener.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/newlistener
na=$(grep -c . configs/jobs_newlistener_a.txt)
nb=$(grep -c . configs/jobs_newlistener_b.txt)
a=$(sbatch --parsable --job-name=nl-a --array=0-$((na - 1))%2 --cpus-per-task=16 --mem=96G \
      --time=03:00:00 scripts/slurm/array.sbatch configs/jobs_newlistener_a.txt)
b=$(sbatch --parsable --job-name=nl-b --array=0-$((nb - 1))%2 --cpus-per-task=16 --mem=96G \
      --time=04:00:00 --dependency=afterany:"$a" \
      scripts/slurm/array.sbatch configs/jobs_newlistener_b.txt)
echo "submitted A=$a ($na tasks) and B=$b ($nb tasks, after A)"
