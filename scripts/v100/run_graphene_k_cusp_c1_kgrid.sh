#!/usr/bin/env bash
# Balanced two-host k-grid convergence subset at fixed lowest finite degauss.
set -euo pipefail

LANE="${1:?usage: run_graphene_k_cusp_c1_kgrid.sh A|B}"
ROOT="${ROOT:-/root/phonon}"
BASE="${BASE:-/data/graphene_k_cusp_c1}"
SUBRUNNER="$ROOT/scripts/v100/run_graphene_k_cusp_c0_dfpt_dense.sh"

case "$LANE" in
    A)
        SPECS=(
            "168|0.993,0.997,1.000"
            "216|1.003,1.007"
        )
        ;;
    B)
        SPECS=(
            "168|1.003,1.007"
            "216|0.993,0.997,1.000"
        )
        ;;
    *)
        echo "unknown lane: $LANE" >&2
        exit 2
        ;;
esac

LANE_ROOT="$BASE/LANE_$LANE"
mkdir -p "$LANE_ROOT"
exec 9>"$LANE_ROOT/.lock"
if ! flock -n 9; then
    echo "[cusp-c1-$LANE] another process is running"
    exit 0
fi
if [[ -s "$LANE_ROOT/DONE" ]]; then
    echo "[cusp-c1-$LANE] already complete"
    exit 0
fi

LOG="$LANE_ROOT/run.log"
exec > >(tee -a "$LOG") 2>&1
echo "[cusp-c1-$LANE] start=$(date -Is)"

for spec in "${SPECS[@]}"; do
    IFS='|' read -r kgrid qpoints <<< "$spec"
    echo "[cusp-c1-$LANE] k=$kgrid q=$qpoints start=$(date -Is)"
    KGRID="$kgrid" \
    QPOINTS_OVERRIDE="$qpoints" \
    WORK_BASE="$BASE/dfpt" \
    WORK="$LANE_ROOT/k$kgrid" \
        /usr/bin/bash "$SUBRUNNER" "$LANE"
    test -s "$LANE_ROOT/k$kgrid/DONE"
    echo "[cusp-c1-$LANE] k=$kgrid complete=$(date -Is)"
done

date -Is > "$LANE_ROOT/COMPLETED_AT"
cp "$LANE_ROOT/COMPLETED_AT" "$LANE_ROOT/DONE.tmp"
mv "$LANE_ROOT/DONE.tmp" "$LANE_ROOT/DONE"
echo "[cusp-c1-$LANE] complete=$(cat "$LANE_ROOT/COMPLETED_AT")"
