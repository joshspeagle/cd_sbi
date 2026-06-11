#!/bin/bash
# Checkpoint POC-suite results to the branch. The remote-execution container is
# ephemeral (reclaimed on inactivity — observed kill 2026-06-10 ~22:21), and
# outputs/ is gitignored, so the branch is the only durable record. Copies the
# small artifacts (parquets, configs, STATUS, seeds) — NOT model.pt — into the
# tracked results_poc/ mirror and pushes. Prints one status line; exits 0 only
# if the mirror is current (no silent-failure || true: rsync was missing in
# this container and a swallowed error here defeats the script's purpose).
set -e
cd /home/user/cd_sbi
mkdir -p results_poc
n=0
for exp in poc_loc_normal_1d poc_sign_normal poc_cauchy poc_gauss_2d_corr; do
  [ -d "outputs/$exp" ] || continue
  while IFS= read -r f; do
    rel="${f#outputs/}"
    mkdir -p "results_poc/$(dirname "$rel")"
    cp -f "$f" "results_poc/$rel"
    n=$((n+1))
  done < <(find "outputs/$exp" -type f \
            \( -name '*.parquet' -o -name 'config.yaml' -o -name 'STATUS' \
               -o -name 'seeds.json' -o -name 'multirun.yaml' \) )
done
cp -f poc_suite.log results_poc/poc_suite_log.txt 2>/dev/null || echo "note: no poc_suite.log yet"
git add results_poc/ > /dev/null
if git diff --cached --quiet; then
  echo "checkpoint: mirror current ($n files), nothing new to commit"
else
  git commit -q -m "Checkpoint POC suite results ($(date -u +%H:%M) UTC)"
  ok=""
  for i in 1 2 3 4; do
    if git push -q -u origin claude/ultracode-effort-Pq4B0 > /dev/null 2>&1; then ok=1; break; fi
    sleep $((2**i))
  done
  if [ -n "$ok" ]; then
    echo "checkpoint: committed+pushed ($n files mirrored)"
  else
    echo "checkpoint: COMMITTED BUT PUSH FAILED — will retry next cycle"
  fi
fi
