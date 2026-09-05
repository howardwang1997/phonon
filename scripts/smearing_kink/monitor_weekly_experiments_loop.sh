#!/usr/bin/env bash
# User-session runner for the weekly experiment monitor.  The remote systemd
# watchdogs own compute recovery; this loop polls, syncs and updates the report.
set -u

PHONON_ROOT="/Users/howardwang/Desktop/playground/phonon"
PHONON_CONDA="/Users/howardwang/miniconda3/bin/conda"
MONITOR="$PHONON_ROOT/scripts/smearing_kink/monitor_weekly_experiments.py"
STATE_DIR="$PHONON_ROOT/results/monitor"
PID_FILE="$STATE_DIR/monitor_loop.pid"

mkdir -p "$STATE_DIR"
exec >> "$STATE_DIR/loop.out" 2>> "$STATE_DIR/loop.err"
if [[ -s "$PID_FILE" ]]; then
    existing_pid="$(<"$PID_FILE")"
    if kill -0 "$existing_pid" 2>/dev/null; then
        echo "monitor loop already running as PID $existing_pid"
        exit 0
    fi
fi

echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT INT TERM
cd "$PHONON_ROOT" || exit 1

while true; do
    echo "[$(date -Iseconds)] monitor pass start"
    if ! "$PHONON_CONDA" --no-plugins run --no-capture-output -n phonon \
        python "$MONITOR"; then
        echo "[$(date -Iseconds)] monitor pass failed; retrying in 300 s"
    fi
    sleep 300
done
