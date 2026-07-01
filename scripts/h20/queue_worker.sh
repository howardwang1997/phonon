#!/usr/bin/env bash
# One GPU worker: pull-and-run until the wave's queue is empty.
# Atomically claims the next job (queue.py), runs it pinned to GPU $1, repeats.
# A failed job keeps its claim for this run (no infinite retry); a fresh
# run_campaign.sh release-stale frees it so the next run retries it.
#   args: GPU_ID WAVE
set -uo pipefail
cd "$HOME/phonon"
G="$1"; WAVE="$2"
export CLAIM_GPU="$G"
PY="$HOME/miniconda3/envs/phonon/bin/python"

while true; do
  line=$("$PY" scripts/h20/hqueue.py claim --wave "$WAVE")
  rc=$?
  [ $rc -ne 0 ] && break          # 1 = nothing claimable, 2 = wave not ready
  IFS=$'\t' read -r JID ENVN GPU GROUP DONE CMD <<<"$line"
  echo "[gpu$G/$WAVE] -> $JID ($GROUP, env=$ENVN) $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$G" CMD="$CMD" \
    bash scripts/h20/run_job.sh "$ENVN" "$GPU" "$GROUP" "$DONE" "$JID" \
    || echo "[gpu$G/$WAVE] job $JID FAILED (continuing) $(date +%H:%M:%S)"
done
echo "[gpu$G/$WAVE] drained $(date +%H:%M:%S)"
