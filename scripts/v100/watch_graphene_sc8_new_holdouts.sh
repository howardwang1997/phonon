#!/usr/bin/env bash
# Restart the experiment-4 tmux only after an abrupt disappearance.  A normal
# success/failure writes a terminal marker and resumes experiment 3 itself.
set -uo pipefail

ROOT=/root/phonon
WORK=/data/graphene_sc8_new_holdouts
PAUSE=/data/phonon_offload/recovery_pause_B
SESSION=graphene_exp4
COUNT_FILE="$WORK/WATCH_RESTART_COUNT"
LOG="$WORK/watch.log"
MAX_RESTARTS=3
mkdir -p "$WORK"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"
}

while [[ ! -e "$WORK/DONE" && ! -e "$WORK/FAILED" ]]; do
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    sleep 60
    continue
  fi
  if pgrep -f "[m]1_1b_graphene_dft.py.*graphene_sc8_dg" >/dev/null; then
    log "8x8 child is still alive outside tmux; leaving it untouched"
    sleep 60
    continue
  fi
  if [[ ! -e "$PAUSE" ]]; then
    log "main session vanished but experiment-3 pause was already released; not re-preempting"
    exit 1
  fi
  restarts=0
  [[ -s "$COUNT_FILE" ]] && read -r restarts < "$COUNT_FILE"
  [[ "$restarts" =~ ^[0-9]+$ ]] || restarts=0
  if (( restarts >= MAX_RESTARTS )); then
    log "restart limit $MAX_RESTARTS reached; releasing experiment 3"
    rm -f "$PAUSE"
    systemctl start phonon-recovery-run@B.service || true
    touch "$WORK/FAILED"
    exit 1
  fi
  restarts=$((restarts + 1))
  printf '%s\n' "$restarts" > "$COUNT_FILE"
  log "experiment-4 session disappeared abruptly; restart $restarts/$MAX_RESTARTS"
  tmux new-session -d -s "$SESSION" \
    "cd $ROOT && bash scripts/v100/run_graphene_sc8_new_holdouts.sh"
  sleep 60
done

log "terminal marker detected; supervisor exiting"
