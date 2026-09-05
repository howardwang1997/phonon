#!/usr/bin/env bash
# Poll the two no-degauss pilot lanes, sync immutable bundles and run S1 audit.
set -euo pipefail

ROOT="${ROOT:-/Users/howardwang/Desktop/playground/phonon}"
CONDA="${CONDA:-/Users/howardwang/miniconda3/bin/conda}"
A_HOST="${A_HOST:-root@100.80.236.112}"
B_HOST="${B_HOST:-root@100.123.220.57}"
JUMP_HOST="${JUMP_HOST:-howardwang@100.105.21.7}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/graphene_k_cusp_nosmear}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-43200}"
QE_TAG="${QE_TAG:-qe75_conda}"
KGRID="${KGRID:-192}"
OUTPUT_LABEL="${OUTPUT_LABEL:-S1_pilot}"
MONITOR_LABEL="${MONITOR_LABEL:-monitor_pilot}"

RESULT_ROOT="$ROOT/results/graphene_k_cusp_nosmear"
RAW_ROOT="$RESULT_ROOT/raw/pilot/k${KGRID}_${QE_TAG}"
OUT="$RESULT_ROOT/$OUTPUT_LABEL"
MONITOR="$RESULT_ROOT/$MONITOR_LABEL"
mkdir -p "$RAW_ROOT/A" "$RAW_ROOT/B" "$OUT" "$MONITOR"

LOCK_DIR="$MONITOR/.lockdir"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "[nosmear-pilot-monitor] another monitor is active"
    exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
if [[ -s "$MONITOR/DONE" ]]; then
    echo "[nosmear-pilot-monitor] already complete"
    exit 0
fi

SSH_OPTIONS=(
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=15
    -o ServerAliveInterval=15
    -o ServerAliveCountMax=2
)

remote_test() {
    local host="$1" path="$2"
    if [[ "$host" == "local" ]]; then
        test -s "$path"
        return
    fi
    if ssh "${SSH_OPTIONS[@]}" "$host" "test -s '$path'"; then
        return 0
    fi
    [[ -n "$JUMP_HOST" && "$JUMP_HOST" != "none" ]] || return 1
    ssh "${SSH_OPTIONS[@]}" -o "ProxyJump=$JUMP_HOST" "$host" "test -s '$path'"
}

sync_lane() {
    local host="$1" remote="$2" destination="$3"
    if [[ "$host" == "local" ]]; then
        cp -pR "$remote/." "$destination/"
        return
    fi
    if scp -q -r "${SSH_OPTIONS[@]}" "$host:$remote/." "$destination/"; then
        return 0
    fi
    [[ -n "$JUMP_HOST" && "$JUMP_HOST" != "none" ]] || return 1
    scp -q -r "${SSH_OPTIONS[@]}" -o "ProxyJump=$JUMP_HOST" \
        "$host:$remote/." "$destination/"
}

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
A_REMOTE="$REMOTE_ROOT/lanes/pilot_A_k${KGRID}_${QE_TAG}"
B_REMOTE="$REMOTE_ROOT/lanes/pilot_B_k${KGRID}_${QE_TAG}"
timestamp > "$MONITOR/STARTED_AT"
waited=0
while true; do
    a_done=0
    b_done=0
    remote_test "$A_HOST" "$A_REMOTE/DONE" && a_done=1 || true
    remote_test "$B_HOST" "$B_REMOTE/DONE" && b_done=1 || true
    printf '%s A_done=%s B_done=%s waited_seconds=%s\n' \
        "$(timestamp)" "$a_done" "$b_done" "$waited" \
        > "$MONITOR/status.txt.tmp"
    mv "$MONITOR/status.txt.tmp" "$MONITOR/status.txt"
    if (( a_done == 1 && b_done == 1 )); then
        break
    fi
    if (( waited >= TIMEOUT_SECONDS )); then
        printf 'timeout after %s seconds\n' "$waited" > "$MONITOR/FAILED"
        exit 3
    fi
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
done

sync_lane "$A_HOST" "$A_REMOTE" "$RAW_ROOT/A"
sync_lane "$B_HOST" "$B_REMOTE" "$RAW_ROOT/B"

set +e
"$CONDA" run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/analyze_graphene_k_cusp_nosmear_pilot.py" \
    --lane-a "$RAW_ROOT/A" --lane-b "$RAW_ROOT/B" --output-dir "$OUT" \
    > "$MONITOR/analysis.log.partial" 2>&1
analysis_status=$?
set -e
mv "$MONITOR/analysis.log.partial" "$MONITOR/analysis.log"
if (( analysis_status != 0 )); then
    printf 'analysis_exit_code=%s\n' "$analysis_status" > "$MONITOR/FAILED"
    exit "$analysis_status"
fi

timestamp > "$MONITOR/COMPLETED_AT"
rm -f "$MONITOR/FAILED"
cp "$MONITOR/COMPLETED_AT" "$MONITOR/DONE.tmp"
mv "$MONITOR/DONE.tmp" "$MONITOR/DONE"
echo "[nosmear-pilot-monitor] complete=$(cat "$MONITOR/COMPLETED_AT")"
