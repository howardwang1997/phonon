#!/usr/bin/env bash
# Autonomous campaign driver: ensures the H20 campaign runs to terminal success
# with no operator intervention. Waits for any in-flight run_campaign.sh, then
# re-launches it (idempotent: done jobs skipped, failed jobs retried via
# release-stale). Stops when the queue reports everything done (Wave B included),
# or when a run makes no progress, or after MAXRUNS.
#
#   nohup bash scripts/h20/auto_driver.sh > results/h20/auto_driver.out 2>&1 & disown
set -uo pipefail
cd "$HOME/phonon"
H=results/h20
PY="$HOME/miniconda3/envs/phonon/bin/python"
MAXRUNS=5
mkdir -p "$H"
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$H/auto_driver.log"; }

log "=== auto_driver start ==="
for i in $(seq 1 $MAXRUNS); do
  # 1. wait for any currently-running campaign (do not double-launch)
  while pgrep -f 'bash scripts/h20/run_campaign.sh' >/dev/null 2>&1; do sleep 30; done
  # 2. snapshot done-count + unfinished-count BEFORE this run
  done_before=$("$PY" scripts/h20/hqueue.py status 2>/dev/null | awk '/^TOTAL/{print $2}')
  unfinished_before=$("$PY" scripts/h20/hqueue.py status 2>/dev/null | awk '/^TOTAL/{print $3-$2}')
  log "run #$i: done_before=$done_before unfinished_before=${unfinished_before:-?}"
  # terminal: nothing unfinished -> campaign complete
  if [ "${unfinished_before:-1}" = "0" ]; then log "all done — exiting"; break; fi
  # 3. launch this run and wait for it
  log "launching run_campaign.sh (run #$i)"
  bash scripts/h20/run_campaign.sh >> "$H/campaign.log" 2>&1 &
  rp=$!; wait "$rp"
  rc=$?
  # 4. snapshot AFTER
  done_after=$("$PY" scripts/h20/hqueue.py status 2>/dev/null | awk '/^TOTAL/{print $2}')
  unfinished_after=$("$PY" scripts/h20/hqueue.py status 2>/dev/null | awk '/^TOTAL/{print $3-$2}')
  log "run #$i exited rc=$rc done=$done_before->$done_after unfinished=${unfinished_before:-?}->${unfinished_after:-?}"
  # terminal: all done now
  if [ "${unfinished_after:-1}" = "0" ]; then log "all green after run #$i — exiting"; break; fi
  # stall guard: no progress in done-count
  if [ "${done_after:-0}" -le "${done_before:-0}" ]; then
    log "STALL: no progress (done $done_before -> $done_after) — need manual fix, exiting"
    break
  fi
done
log "=== auto_driver end ==="
