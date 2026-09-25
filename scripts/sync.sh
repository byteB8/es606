#!/usr/bin/env bash
# Sync this repo with its remote copy (~/codes/eg606) and run commands there.
#
#   scripts/sync.sh push                 # local -> remote (local is the source of truth)
#   scripts/sync.sh pull                 # remote results/ and logs/ -> local
#   scripts/sync.sh run  "<cmd>"         # run in the remote repo dir, foreground
#   scripts/sync.sh bg   <name> "<cmd>"  # detached tmux session, log to logs/<name>.log (not on HPC)
#   scripts/sync.sh sbatch "<cmd>"       # submit a Slurm job (Singularity HPC)
#   scripts/sync.sh queue                # squeue for your jobs
#   scripts/sync.sh status               # full read-only status (works on the HPC too)
#
# Server: bhaskar by default; EG606_SERVER=ramanujan to switch.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Machine addresses live in scripts/servers.env (untracked). See servers.env.example.
[ -f "$here/scripts/servers.env" ] && . "$here/scripts/servers.env"
case "${EG606_SERVER:-bhaskar}" in
  bhaskar)   host=${BHASKAR_HOST:?set it in scripts/servers.env}; port=${BHASKAR_PORT:-22} ;;
  ramanujan) host=${RAMANUJAN_HOST:?set it in scripts/servers.env}; port=${RAMANUJAN_PORT:-22} ;;
  # Singularity HPC: batch only, never run work directly over ssh (cluster policy)
  singularity) host=${SINGULARITY_HOST:?set it in scripts/servers.env}; port=${SINGULARITY_PORT:-22} ;;
  *) echo "unknown EG606_SERVER=$EG606_SERVER" >&2; exit 1 ;;
esac
# ramanujan keeps its own env in ~/wrf, so sync into a subfolder: an rsync --delete straight
# into ~/wrf would remove wrfEnv.
if [ "${EG606_SERVER:-bhaskar}" = ramanujan ]; then remote_dir=wrf/runs; else remote_dir=codes/eg606; fi
ssh_cmd="ssh -p $port -o BatchMode=yes"
# Runtime folders are excluded, which also protects them from --delete on the remote.
excludes=(--exclude .git/ --exclude .venv/ --exclude __pycache__/ --exclude '*.egg-info/'
          --exclude results/ --exclude logs/ --exclude outputs/)

case "${1:-}" in
  push)
    $ssh_cmd "$host" "mkdir -p $remote_dir"
    rsync -az --delete "${excludes[@]}" -e "$ssh_cmd" "$here/" "$host:$remote_dir/" ;;
  pull)
    rsync -az --ignore-missing-args -e "$ssh_cmd" \
      "$host:$remote_dir/results" "$host:$remote_dir/logs" "$here/" ;;
  run)
    shift
    if [ "${EG606_SERVER:-bhaskar}" = singularity ]; then
      echo "refusing: Singularity policy forbids interactive jobs (accounts are blocked for a week)." >&2
      echo "Submit with: scripts/sync.sh sbatch \"<command>\"" >&2; exit 1
    fi
    $ssh_cmd "$host" "cd $remote_dir && $*" ;;
  sbatch)
    shift
    $ssh_cmd "$host" "cd $remote_dir && mkdir -p logs && sbatch scripts/slurm/train.sbatch '$*'" ;;
  queue)
    $ssh_cmd "$host" "squeue -u \$(whoami)" ;;
  status)
    # Read-only monitoring, allowed everywhere: it inspects the queue and log files, it is not a job.
    $ssh_cmd "$host" "cd $remote_dir && bash scripts/hpc_status.sh" ;;
  wait-for)
    # tmux matches -t targets by PREFIX, so "-t pipeline" also matches "pipeline2" and a job can
    # end up waiting for itself. "=name" forces an exact match.
    shift
    $ssh_cmd "$host" "while tmux has-session -t=$1 2>/dev/null; do sleep 30; done" ;;
  bg)
    if [ "${EG606_SERVER:-bhaskar}" = singularity ]; then
      echo "refusing: no tmux jobs on the HPC cluster; use scripts/sync.sh sbatch \"<command>\"" >&2; exit 1
    fi
    name="$2"; shift 2
    # Ship the command as a script file so quoting and && chains survive intact.
    printf '%s\n' "$*" | $ssh_cmd "$host" "cd $remote_dir && mkdir -p logs && cat > logs/$name.sh"
    $ssh_cmd "$host" "cd $remote_dir && tmux new-session -d -s '$name' 'bash logs/$name.sh 2>&1 | tee logs/$name.log'"
    echo "started tmux session '$name' on ${EG606_SERVER:-bhaskar}; log: $remote_dir/logs/$name.log" ;;
  *)
    echo "usage: $0 push | pull | run <cmd> | bg <name> <cmd> | sbatch <cmd> | queue | status" >&2; exit 1 ;;
esac
