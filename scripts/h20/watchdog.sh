#!/usr/bin/env bash
# Autonomous watchdog (robust): snapshots campaign status every 60s and reports
# CURRENT-run failures only. A "current failure" = a `FAIL rc=` line in the
# campaign.log written by this run (truncated at launch). For each newly-seen
# (jobid, mtime) it dumps the joblog tail into results/h20/failures.log so the
# autonomous agent can self-heal. Re-run/truncated logs get a new mtime -> re-reported.
#   nohup bash scripts/h20/watchdog.sh > results/h20/watchdog.out 2>&1 & disown
set -uo pipefail
cd "$HOME/phonon"
H=results/h20
mkdir -p "$H/joblogs"
SEENF="$H/watchdog_seen.txt"; touch "$SEENF"
while true; do
  ts=$(date +%H:%M:%S)
  if [ -f "$H/jobs.jsonl" ]; then
    fail_n=$(grep -cE 'FAIL rc=' "$H/campaign.log" 2>/dev/null || echo 0)
    e1d=$(ls "$H"/e1/*.csv 2>/dev/null | wc -l)
    echo "$ts e1_csv~$e1d fails~$fail_n" >> "$H/watchdog.log"
  fi
  # current-run failures: lines like  [gpuN/A] job JOBID FAILED (continuing) ...
  grep -E 'job [A-Za-z0-9_.-]+ FAILED \(continuing\)' "$H/campaign.log" 2>/dev/null \
    | sed -E 's/.*job ([A-Za-z0-9_.-]+) FAILED.*/\1/' | sort -u | while read -r jid; do
      [ -z "$jid" ] && continue
      jl="$H/joblogs/$jid.log"; [ -f "$jl" ] || continue
      mt=$(stat -c %Y "$jl" 2>/dev/null || echo 0)
      key="$jid $mt"
      grep -qF "$key" "$SEENF" 2>/dev/null && continue
      echo "$key" >> "$SEENF"
      { echo "===== FAIL $jid @ $ts (log mtime $mt) ====="
        grep -E 'Error|Traceback|cannot|No (module|such|file)|Import|assert|CUDA|RuntimeError|ValueError|KeyError|Mace|Exception' "$jl" | tail -10
      } >> "$H/failures.log"
  done
  sleep 60
done
