#!/usr/bin/env bash
# Reuse the completed graphene EPW coarse Wannier representation without
# modifying /data/graphene_epw3.  This is an E0 feasibility restart, not a
# quantitative replacement for the direct low-smearing DFPT line.
set -euo pipefail

MODE="${1:-production}"
ROOT="${ROOT:-/root/phonon}"
SOURCE="${SOURCE:-/data/graphene_epw3}"
BASE="${BASE:-/data/graphene_e0_epw_pilot}"

case "$MODE" in
    smoke)
        GRID=18
        WORK_NAME="k18_T450"
        INPUT="$ROOT/configs/graphene_e0_epw_pilot/epw_k18_T450.in"
        QPOINTS="$ROOT/configs/graphene_e0_epw_pilot/qpoints_GK5_k18.dat"
        ;;
    production)
        GRID=360
        WORK_NAME="k360_T450"
        INPUT="$ROOT/configs/graphene_e0_epw_pilot/epw_k360_T450.in"
        QPOINTS="$ROOT/configs/graphene_e0_epw_pilot/qpoints_GK9_k360.dat"
        ;;
    convergence)
        GRID=720
        WORK_NAME="k720_T450"
        INPUT="$ROOT/configs/graphene_e0_epw_pilot/epw_k720_T450.in"
        QPOINTS="$ROOT/configs/graphene_e0_epw_pilot/qpoints_GK9_k360.dat"
        ;;
    static)
        GRID=720
        WORK_NAME="k720_static_T450"
        INPUT="$ROOT/configs/graphene_e0_epw_pilot/epw_k720_static_T450.in"
        QPOINTS="$ROOT/configs/graphene_e0_epw_pilot/qpoints_GK9_k360.dat"
        ;;
    static360)
        GRID=360
        WORK_NAME="k360_static_T450"
        INPUT="$ROOT/configs/graphene_e0_epw_pilot/epw_k360_static_T450.in"
        QPOINTS="$ROOT/configs/graphene_e0_epw_pilot/qpoints_GK9_k360.dat"
        ;;
    static005)
        GRID=720
        WORK_NAME="k720_static005_T450"
        INPUT="$ROOT/configs/graphene_e0_epw_pilot/epw_k720_static005_T450.in"
        QPOINTS="$ROOT/configs/graphene_e0_epw_pilot/qpoints_GK9_k360.dat"
        ;;
    *)
        echo "usage: $0 smoke|production|convergence|static|static360|static005" >&2
        exit 2
        ;;
esac

WORK="$BASE/$WORK_NAME"
mkdir -p "$WORK/tmp"
if [[ -e "$WORK/DONE" ]]; then
    echo "E0 EPW $MODE already complete"
    exit 0
fi

for source in \
    "$SOURCE/epwdata.fmt" \
    "$SOURCE/dmedata.fmt" \
    "$SOURCE/vmedata.fmt" \
    "$SOURCE/crystal.fmt" \
    "$SOURCE/decay.H" \
    "$SOURCE/decay.P" \
    "$SOURCE/decay.dynmat" \
    "$SOURCE/decay.epmate" \
    "$SOURCE/decay.epmatp" \
    "$SOURCE/decay.r" \
    "$SOURCE/decay.v" \
    "$SOURCE/graphene.ukk" \
    "$SOURCE/graphene.kmap" \
    "$SOURCE/graphene.kgmap"; do
    test -s "$source"
    destination="$WORK/$(basename "$source")"
    if [[ ! -s "$destination" ]]; then
        cp -p "$source" "$destination"
    fi
done
if [[ ! -s "$WORK/tmp/graphene.epmatwp" ]]; then
    cp -p "$SOURCE/tmp/graphene.epmatwp" "$WORK/tmp/graphene.epmatwp"
fi
cp -p "$INPUT" "$WORK/epw.in"
cp -p "$QPOINTS" "$WORK/qpoints.dat"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate phonon
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$WORK"

start="$(date -Is)"
echo "=== graphene E0 EPW restart mode=$MODE grid=$GRID start=$start ==="
set +e
mpirun --allow-run-as-root -np 1 epw.x -in epw.in > epw.out.partial 2>&1
status=$?
set -e
mv epw.out.partial epw.out
if [[ "$status" -ne 0 ]]; then
    echo "EPW restart failed with exit=$status" >&2
    exit "$status"
fi
if ! grep -q "Total program execution" epw.out; then
    echo "EPW returned zero but completion marker is absent" >&2
    exit 3
fi
printf '%s\n' "$start" > STARTED_AT
date -Is > COMPLETED_AT
touch DONE
echo "=== graphene E0 EPW restart mode=$MODE grid=$GRID complete=$(date -Is) ==="
