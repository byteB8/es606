#!/usr/bin/env bash
# Copy the OpenMIIR raw EEG to the NAS, surviving flaky servers.
#
# Both institute machines have been dropping SSH mid-transfer (bhaskar's sshd stops answering,
# ramanujan sits at load 70+), so this retries, resumes partial files rather than restarting them,
# and falls back to whichever host answers. The NAS is the same filesystem either way.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$here/scripts/servers.env"
src="${1:-/mnt/bdrv/es606/data/openmiir/OpenMIIR-RawEEG_v1/}"
attempts="${ATTEMPTS:-40}"

try() {  # host port remote_root
  rsync -a --partial --append-verify --info=stats1 --timeout=180 \
    -e "ssh -p $2 -o BatchMode=yes -o ConnectTimeout=60 -o ServerAliveInterval=20 -o ServerAliveCountMax=3" \
    "$src" "$1:$3/balbir/egdta/raw/openmiir/eeg/"
}

for i in $(seq 1 "$attempts"); do
  for spec in "$RAMANUJAN_HOST ${RAMANUJAN_PORT:-2022} /mnt/nas_ramanujan" \
              "$BHASKAR_HOST ${BHASKAR_PORT:-22} /mnt/nas"; do
    read -r host port root <<<"$spec"
    echo "[attempt $i] $host:$root"
    ssh -p "$port" -o BatchMode=yes -o ConnectTimeout=60 "$host" \
        "mkdir -p $root/balbir/egdta/raw/openmiir/eeg" 2>/dev/null || { echo "  unreachable"; continue; }
    if try "$host" "$port" "$root"; then
      echo "TRANSFER COMPLETE via $host"
      exit 0
    fi
    echo "  interrupted, will resume"
  done
  sleep 30
done
echo "GAVE UP after $attempts attempts"
exit 1
