#!/usr/bin/env bash
# Poll the two k240 d=0.003 diagnostics, sync bundles and decide on k288.
set -euo pipefail

ROOT="${ROOT:-/Users/howardwang/Desktop/playground/phonon}"
CONDA="${CONDA:-/Users/howardwang/miniconda3/bin/conda}"
A_HOST="${A_HOST:-root@100.80.236.112}"
B_HOST="${B_HOST:-root@100.123.220.57}"
JUMP_HOST="${JUMP_HOST:-howardwang@100.105.21.7}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/graphene_k_cusp_nosmear}"
POLL_SECONDS="${POLL_SECONDS:-180}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-64800}"

RESULT_ROOT="$ROOT/results/graphene_k_cusp_nosmear"
RAW_ROOT="$RESULT_ROOT/raw/diagnostic_d003/k240_qe75_npk120k"
A_NAME="diagnostic_A_k240_qe75_npk120k"
B_NAME="diagnostic_B_k240_qe75_npk120k"
A_DEST="$RAW_ROOT/$A_NAME"
B_DEST="$RAW_ROOT/$B_NAME"
OUT="$RESULT_ROOT/S4a_d003_kgrid"
MONITOR="$RESULT_ROOT/monitor_d003_kgrid"
mkdir -p "$RAW_ROOT" "$OUT" "$MONITOR"

LOCK_DIR="$MONITOR/.lockdir"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "[nosmear-d003-monitor] another monitor is active"
    exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
if [[ -s "$MONITOR/DONE" ]]; then
    echo "[nosmear-d003-monitor] already complete"
    exit 0
fi

SSH_OPTIONS=(
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=15
    -o ServerAliveInterval=15
    -o ServerAliveCountMax=2
)

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

remote_state() {
    local host="$1" path="$2" response
    if response="$(ssh "${SSH_OPTIONS[@]}" "$host" \
        "if test -s '$path'; then printf DONE; else printf WAITING; fi" 2>/dev/null)"; then
        printf '%s\n' "$response"
        return 0
    fi
    if [[ -n "$JUMP_HOST" && "$JUMP_HOST" != "none" ]] \
        && response="$(ssh "${SSH_OPTIONS[@]}" -o "ProxyJump=$JUMP_HOST" "$host" \
            "if test -s '$path'; then printf DONE; else printf WAITING; fi" 2>/dev/null)"; then
        printf '%s\n' "$response"
        return 0
    fi
    printf 'UNREACHABLE\n'
}

sync_lane() {
    local host="$1" remote="$2" destination="$3" attempt
    mkdir -p "$destination"
    for attempt in $(seq 1 20); do
        if scp -q -r "${SSH_OPTIONS[@]}" "$host:$remote/." "$destination/"; then
            test -s "$destination/DONE"
            test -s "$destination/manifest.json"
            test -s "$destination/dfpt_points.csv"
            return 0
        fi
        if [[ -n "$JUMP_HOST" && "$JUMP_HOST" != "none" ]] \
            && scp -q -r "${SSH_OPTIONS[@]}" -o "ProxyJump=$JUMP_HOST" \
                "$host:$remote/." "$destination/"; then
            test -s "$destination/DONE"
            test -s "$destination/manifest.json"
            test -s "$destination/dfpt_points.csv"
            return 0
        fi
        printf '%s sync_retry host=%s attempt=%s\n' \
            "$(timestamp)" "$host" "$attempt" >> "$MONITOR/sync.log"
        sleep 60
    done
    return 1
}

A_REMOTE="$REMOTE_ROOT/lanes/$A_NAME"
B_REMOTE="$REMOTE_ROOT/lanes/$B_NAME"
timestamp > "$MONITOR/STARTED_AT"
waited=0
while true; do
    a_state="$(remote_state "$A_HOST" "$A_REMOTE/DONE")"
    b_state="$(remote_state "$B_HOST" "$B_REMOTE/DONE")"
    printf '%s A=%s B=%s waited_seconds=%s\n' \
        "$(timestamp)" "$a_state" "$b_state" "$waited" \
        > "$MONITOR/status.txt.tmp"
    mv "$MONITOR/status.txt.tmp" "$MONITOR/status.txt"
    if [[ "$a_state" == "DONE" && "$b_state" == "DONE" ]]; then
        break
    fi
    if (( waited >= TIMEOUT_SECONDS )); then
        printf 'timeout after %s seconds\n' "$waited" > "$MONITOR/FAILED"
        exit 3
    fi
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
done

sync_lane "$A_HOST" "$A_REMOTE" "$A_DEST"
sync_lane "$B_HOST" "$B_REMOTE" "$B_DEST"

CONVERGENCE="$RESULT_ROOT/raw/convergence/k240_qe75_npk120k"
set +e
"$CONDA" run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/analyze_graphene_k_cusp_nosmear_d003_kgrid.py" \
    --k192-holdout-csv "$RESULT_ROOT/S4_holdout/holdout_predictions.csv" \
    --k240-source "$CONVERGENCE/convergence_A_k240_qe75_npk120k" \
    --k240-source "$CONVERGENCE/convergence_B_k240_qe75_npk120k" \
    --k240-source "$A_DEST" --k240-source "$B_DEST" \
    --release "$RESULT_ROOT/S4_holdout/RELEASE_D003_KGRID" \
    --output-dir "$OUT" > "$MONITOR/analysis.log.partial" 2>&1
analysis_status=$?
set -e
mv "$MONITOR/analysis.log.partial" "$MONITOR/analysis.log"
if (( analysis_status != 0 && analysis_status != 3 )); then
    printf 'analysis_exit_code=%s\n' "$analysis_status" > "$MONITOR/FAILED"
    exit "$analysis_status"
fi

if (( analysis_status == 0 )); then
    printf 'k192_d003_converged_model_shape_failure\n' > "$MONITOR/RESULT"
else
    printf 'requires_k288_d003\n' > "$MONITOR/RESULT"
fi
timestamp > "$MONITOR/COMPLETED_AT"
rm -f "$MONITOR/FAILED"
cp "$MONITOR/COMPLETED_AT" "$MONITOR/DONE.tmp"
mv "$MONITOR/DONE.tmp" "$MONITOR/DONE"
echo "[nosmear-d003-monitor] result=$(cat "$MONITOR/RESULT") complete=$(cat "$MONITOR/COMPLETED_AT")"
