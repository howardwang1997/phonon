#!/usr/bin/env bash
# Collect both C0 DFPT lanes over Tailscale and run the frozen local gate.
set -euo pipefail

ROOT="${ROOT:-/Users/howardwang/Desktop/playground/phonon}"
CONDA="${CONDA:-/Users/howardwang/miniconda3/bin/conda}"
A_HOST="${A_HOST:-root@100.80.236.112}"
B_HOST="${B_HOST:-root@100.123.220.57}"
A_REMOTE="${A_REMOTE:-/data/graphene_k_cusp_c0/FD0_K_DENSE_A}"
B_REMOTE="${B_REMOTE:-/data/graphene_k_cusp_c0/FD0_K_DENSE_B}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-172800}"

RESULT_ROOT="$ROOT/results/graphene_kohn_cusp_two_methods"
COLLECT_ROOT="$RESULT_ROOT/C0_remote_dfpt"
ANALYSIS_ROOT="$RESULT_ROOT/C0_fd0_k192_dense"
MONITOR_ROOT="$RESULT_ROOT/C0_monitor"
mkdir -p "$COLLECT_ROOT/A" "$COLLECT_ROOT/B" "$ANALYSIS_ROOT" "$MONITOR_ROOT"

LOCK_DIR="$MONITOR_ROOT/.lockdir"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "[cusp-c0-monitor] another monitor is running"
    exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

if [[ -s "$MONITOR_ROOT/DONE" ]]; then
    echo "[cusp-c0-monitor] already complete"
    exit 0
fi

SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)

remote_done() {
    local host="$1"
    local directory="$2"
    ssh "${SSH_OPTIONS[@]}" "$host" "test -s '$directory/DONE'"
}

timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

timestamp > "$MONITOR_ROOT/STARTED_AT"
waited=0
while true; do
    a_done=0
    b_done=0
    if remote_done "$A_HOST" "$A_REMOTE"; then
        a_done=1
    fi
    if remote_done "$B_HOST" "$B_REMOTE"; then
        b_done=1
    fi
    printf '%s A_done=%s B_done=%s waited_seconds=%s\n' \
        "$(timestamp)" "$a_done" "$b_done" "$waited" \
        > "$MONITOR_ROOT/status.txt.tmp"
    mv "$MONITOR_ROOT/status.txt.tmp" "$MONITOR_ROOT/status.txt"
    if (( a_done == 1 && b_done == 1 )); then
        break
    fi
    if (( waited >= TIMEOUT_SECONDS )); then
        printf '%s\n' "timeout after ${waited}s" > "$MONITOR_ROOT/FAILED"
        exit 3
    fi
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
done

scp "${SSH_OPTIONS[@]}" \
    "$A_HOST:$A_REMOTE/graphene_FD0_K_DENSE_A_dfpt.csv" \
    "$A_HOST:$A_REMOTE/manifest.json" \
    "$A_HOST:$A_REMOTE/run.log" \
    "$A_HOST:$A_REMOTE/COMPLETED_AT" \
    "$A_HOST:$A_REMOTE/DONE" \
    "$COLLECT_ROOT/A/"
scp "${SSH_OPTIONS[@]}" \
    "$B_HOST:$B_REMOTE/graphene_FD0_K_DENSE_B_dfpt.csv" \
    "$B_HOST:$B_REMOTE/manifest.json" \
    "$B_HOST:$B_REMOTE/run.log" \
    "$B_HOST:$B_REMOTE/COMPLETED_AT" \
    "$B_HOST:$B_REMOTE/DONE" \
    "$COLLECT_ROOT/B/"

set +e
"$CONDA" run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/analyze_graphene_k_cusp_c0_dfpt_dense.py" \
    --lane-a "$COLLECT_ROOT/A/graphene_FD0_K_DENSE_A_dfpt.csv" \
    --lane-b "$COLLECT_ROOT/B/graphene_FD0_K_DENSE_B_dfpt.csv" \
    --output-dir "$ANALYSIS_ROOT" \
    > "$MONITOR_ROOT/analysis.log.partial" 2>&1
analysis_status=$?
set -e
mv "$MONITOR_ROOT/analysis.log.partial" "$MONITOR_ROOT/analysis.log"

if (( analysis_status != 0 )); then
    printf 'analysis_exit_code=%s\n' "$analysis_status" > "$MONITOR_ROOT/FAILED"
    exit "$analysis_status"
fi

timestamp > "$MONITOR_ROOT/COMPLETED_AT"
cp "$MONITOR_ROOT/COMPLETED_AT" "$MONITOR_ROOT/DONE.tmp"
mv "$MONITOR_ROOT/DONE.tmp" "$MONITOR_ROOT/DONE"
echo "[cusp-c0-monitor] complete=$(cat "$MONITOR_ROOT/COMPLETED_AT")"
