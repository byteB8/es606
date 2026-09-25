#!/usr/bin/env bash
# Publish the code to GitHub with a clean, separate history.
#
# The working repository carries an internal record (LAB_NOTEBOOK.md, results/, raw logs) and the
# history of those files. This copies only the tracked source at HEAD, minus the internal parts,
# into a sibling repository whose history contains nothing else, and pushes that.
#
#   scripts/publish.sh "message"
set -euo pipefail
msg="${1:-Update}"
root="$(git rev-parse --show-toplevel)"
pub="${EG606_PUBLISH_DIR:-$(dirname "$root")/es606-publish}"

mkdir -p "$pub"
[ -d "$pub/.git" ] || git -C "$pub" init -q -b main
find "$pub" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
git -C "$root" archive HEAD | tar -x -C "$pub"      # tracked files only
rm -rf "$pub/LAB_NOTEBOOK.md" "$pub/results" "$pub/scripts/servers.env"

cd "$pub"
git add -A
if git diff --cached --quiet; then
  echo "nothing to publish"; exit 0
fi
git commit -q -m "$msg"
git push -q origin main 2>/dev/null || echo "no remote yet: run  gh repo create es606 --private --source=$pub --push"
echo "published $(git rev-list --count main) commit(s) from $pub"
