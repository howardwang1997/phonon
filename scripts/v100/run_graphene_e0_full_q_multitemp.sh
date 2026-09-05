#!/usr/bin/env bash
# Export q6 static self energies at 300 and 600 K after both sparse gates pass.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
SOURCE="${SOURCE:-/data/graphene_e0_epw_matched/k18_q9_ex1_pifroz}"
RELEASE="$SOURCE/analysis_multitemp/MULTITEMP_SPECTRAL_GATE_PASS"
WORK_ROOT="$SOURCE/full_q6_multitemp"

test -s "$RELEASE"
mkdir -p "$WORK_ROOT"
exec 9>"$WORK_ROOT/.lock"
if ! flock -n 9; then
    echo "[e0-full-q-multitemp] another process is already running"
    exit 0
fi
if [[ -s "$WORK_ROOT/DONE" ]]; then
    echo "[e0-full-q-multitemp] already complete"
    exit 0
fi

run_temperature() {
    local temperature="$1" work runner
    work="$SOURCE/full_q6_${temperature}K"
    runner="$ROOT/scripts/v100/run_graphene_e0_full_q_gate.sh"
    env TARGET_TEMPERATURE_K="$temperature" WORK="$work" \
        RELEASE_MARKER="$RELEASE" ROOT="$ROOT" SOURCE="$SOURCE" \
        /usr/bin/bash "$runner"
}

run_temperature 300 &
pid_300=$!
run_temperature 600 &
pid_600=$!
parallel_status=0
wait "$pid_300" || parallel_status=1
wait "$pid_600" || parallel_status=1
[[ "$parallel_status" -eq 0 ]]

for temperature in 300 450 600; do
    test -s "$SOURCE/full_q6_${temperature}K/DONE"
    test -s "$SOURCE/full_q6_${temperature}K/audit/FULL_Q_DATA_READY"
done
date -Is > "$WORK_ROOT/DONE.tmp"
mv "$WORK_ROOT/DONE.tmp" "$WORK_ROOT/DONE"
echo "[e0-full-q-multitemp] 300/450/600 K q6 grids complete=$(date -Is)"
