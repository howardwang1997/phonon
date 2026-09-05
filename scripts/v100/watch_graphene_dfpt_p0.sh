#!/usr/bin/env bash
# Lightweight tmux supervisor for the restartable P0 DFPT line-cut job.  It
# never kills a live calculation; it only restarts a lane after its tmux/process
# has disappeared without producing DONE.
set -uo pipefail

LANE="${1:?usage: bash scripts/v100/watch_graphene_dfpt_p0.sh A|B}"
case "$LANE" in
  A)
    QEBIN=/root/miniconda3/envs/qe/bin
    EXPECTED_LINES=15
    ;;
  B)
    QEBIN=/root/miniconda3/envs/phonon/bin
    EXPECTED_LINES=43
    ;;
  *)
    echo "unknown lane '$LANE'" >&2
    exit 2
    ;;
esac

ROOT=/root/phonon
WORK="/data/p0_graphene_dfpt/$LANE"
SESSION="p0_dfpt_$LANE"
COUNT_FILE="$WORK/RESTART_COUNT"
WATCH_LOG="$WORK/watch.log"
MAX_RESTARTS=5
mkdir -p "$WORK"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$WATCH_LOG"
}

valid_done() {
  local csv="$WORK/dfpt_linecuts_${LANE}.csv" lines
  [[ -e "$WORK/DONE" && -s "$csv" ]] || return 1
  lines="$(wc -l < "$csv")"
  [[ "$lines" =~ ^[0-9]+$ ]] && (( lines == EXPECTED_LINES ))
}

while ! valid_done; do
  if [[ -e "$WORK/DONE" ]]; then
    invalid="$WORK/DONE.invalid.$(date +%s)"
    mv "$WORK/DONE" "$invalid"
    log "archived invalid DONE marker as $invalid; complete CSV is missing"
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    sleep 60
    continue
  fi
  if pgrep -f "[r]un_graphene_dfpt_linecuts_p0.sh $LANE" >/dev/null; then
    log "lane process is alive outside tmux; leaving it untouched"
    sleep 60
    continue
  fi

  restarts=0
  if [[ -s "$COUNT_FILE" ]]; then
    read -r restarts < "$COUNT_FILE"
  fi
  if [[ ! "$restarts" =~ ^[0-9]+$ ]]; then
    restarts=0
  fi
  if (( restarts >= MAX_RESTARTS )); then
    log "giving up after $restarts restarts; manual inspection required"
    touch "$WORK/MONITOR_FAILED"
    exit 1
  fi
  restarts=$((restarts + 1))
  printf '%s\n' "$restarts" > "$COUNT_FILE"
  log "main lane vanished before DONE; starting restart $restarts/$MAX_RESTARTS"
  tmux new-session -d -s "$SESSION" \
    "cd $ROOT && QEBIN=$QEBIN NP=8 bash scripts/v100/run_graphene_dfpt_linecuts_p0.sh $LANE"
  sleep 60
done

log "DONE detected; supervisor exiting"
