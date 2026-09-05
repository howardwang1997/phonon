#!/usr/bin/env bash
# Durable supervisor for one graphene q-space DFPT campaign.
set -uo pipefail

CAMPAIGN="${1:?usage: bash scripts/v100/watch_graphene_qspace_dfpt.sh DEV_A|DEV_B|HOLD_A|HOLD_B}"
case "$CAMPAIGN" in
  DEV_A)
    EXPECTED_LINES=15
    QEBIN=/root/miniconda3/envs/qe/bin
    ;;
  DEV_B)
    EXPECTED_LINES=15
    QEBIN=/root/miniconda3/envs/phonon/bin
    ;;
  HOLD_A)
    EXPECTED_LINES=29
    QEBIN=/root/miniconda3/envs/qe/bin
    ;;
  HOLD_B)
    EXPECTED_LINES=20
    QEBIN=/root/miniconda3/envs/phonon/bin
    ;;
  *)
    echo "unknown campaign '$CAMPAIGN'" >&2
    exit 2
    ;;
esac

ROOT=/root/phonon
WORK="/data/graphene_qspace_dfpt/$CAMPAIGN"
SESSION="qspace_dfpt_$CAMPAIGN"
COUNT_FILE="$WORK/RESTART_COUNT"
WATCH_LOG="$WORK/watch.log"
MAX_RESTARTS=5
mkdir -p "$WORK"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$WATCH_LOG"
}

valid_done() {
  local csv="$WORK/dfpt_${CAMPAIGN}.csv" lines
  [[ -e "$WORK/DONE" && -s "$csv" ]] || return 1
  lines="$(wc -l < "$csv")"
  [[ "$lines" =~ ^[0-9]+$ ]] && (( lines == EXPECTED_LINES ))
}

while ! valid_done; do
  if [[ -e "$WORK/DONE" ]]; then
    invalid="$WORK/DONE.invalid.$(date +%s)"
    mv "$WORK/DONE" "$invalid"
    log "archived invalid DONE marker as $invalid"
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    sleep 60
    continue
  fi
  if pgrep -f "[r]un_graphene_qspace_dfpt.sh $CAMPAIGN" >/dev/null; then
    log "campaign process is alive outside tmux; leaving it untouched"
    sleep 60
    continue
  fi

  restarts=0
  if [[ -s "$COUNT_FILE" ]]; then
    read -r restarts < "$COUNT_FILE"
  fi
  [[ "$restarts" =~ ^[0-9]+$ ]] || restarts=0
  if (( restarts >= MAX_RESTARTS )); then
    log "giving up after $restarts restarts; manual inspection required"
    touch "$WORK/MONITOR_FAILED"
    exit 1
  fi
  restarts=$((restarts + 1))
  printf '%s\n' "$restarts" > "$COUNT_FILE"
  log "main campaign vanished before valid DONE; starting restart $restarts/$MAX_RESTARTS"
  tmux new-session -d -s "$SESSION" \
    "cd $ROOT && QEBIN=$QEBIN NP=8 bash scripts/v100/run_graphene_qspace_dfpt.sh $CAMPAIGN"
  sleep 60
done

log "valid DONE detected; supervisor exiting"
