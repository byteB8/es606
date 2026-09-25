#!/usr/bin/env bash
# One-shot status of our work on the Singularity cluster. Run from ~/codes/eg606 on the cluster,
# or locally as:  EG606_SERVER=singularity scripts/sync.sh run 'bash scripts/hpc_status.sh'
if ! command -v squeue >/dev/null; then            # bhaskar / ramanujan have no scheduler
  echo "=== tmux sessions"; tmux ls 2>/dev/null || echo "  none"
  echo; echo "=== GPUs"; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  echo; echo "=== latest logs"; for f in $(ls -t logs/*.log 2>/dev/null | head -3); do echo "--- $f"; tail -3 "$f"; done
  exit 0
fi

echo "=== my jobs (R = running, PD = pending)"
squeue -u "$(whoami)" -o "%.8i %.14j %.2t %.11M %.11L %.4C %.14b %R" || true

echo; echo "=== cluster load"
sinfo -o "%.10P %.6D %.14G %.20N %.10T"
squeue -h -o "%t" | sort | uniq -c | awk '{printf "  %s jobs %s\n", $1, $2}'

echo; echo "=== my jobs today (finished ones too)"
sacct -u "$(whoami)" --starttime today --format=JobID%10,JobName%18,State%12,Elapsed%10,MaxRSS%8 2>/dev/null | head -20

echo; echo "=== GPU usage of my running jobs (sampled every 30 s by the job itself)"
for f in $(ls -t logs/gpu-*.log 2>/dev/null | head -3); do
  id=$(basename "$f" .log | cut -d- -f2)
  if squeue -h -j "$id" >/dev/null 2>&1 && [ -n "$(squeue -h -j "$id" 2>/dev/null)" ]; then
    echo "--- job $id (latest sample)"; tail -2 "$f"
  fi
done

echo; echo "=== latest output lines"
for f in $(ls -t logs/*.out 2>/dev/null | head -3); do
  echo "--- $f"; tail -3 "$f"
done
