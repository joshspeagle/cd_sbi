#!/bin/bash
# Checkpoint POC-suite results to the branch. The remote-execution container is
# ephemeral (reclaimed on inactivity — observed kill 2026-06-10 ~22:21), and
# outputs/ is gitignored, so the branch is the only durable record. Copies the
# small artifacts (parquets, configs, STATUS, seeds) — NOT model.pt — into the
# tracked results_poc/ mirror and pushes. Quiet: prints nothing on success.
set -e
cd /home/user/cd_sbi
mkdir -p results_poc
rsync -a --prune-empty-dirs \
  --include='*/' \
  --include='*.parquet' \
  --include='config.yaml' \
  --include='STATUS' \
  --include='seeds.json' \
  --include='multirun.yaml' \
  --exclude='*' \
  outputs/poc_loc_normal_1d outputs/poc_sign_normal outputs/poc_cauchy outputs/poc_gauss_2d_corr \
  results_poc/ 2>/dev/null || true
cp -f poc_suite.log results_poc/ 2>/dev/null || true
git add results_poc/ > /dev/null 2>&1
if ! git diff --cached --quiet; then
  git commit -q -m "Checkpoint POC suite results ($(date -u +%H:%M) UTC)" > /dev/null 2>&1
  for i in 1 2 3 4; do
    git push -q -u origin claude/ultracode-effort-Pq4B0 > /dev/null 2>&1 && break
    sleep $((2**i))
  done
fi
