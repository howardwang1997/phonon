#!/usr/bin/env bash
# Second, non-overlapping physical-FD label wave for the finite-temperature
# graphene MLIP.  Together with wave 1 this gives 24 train + 6 validation
# structures at each temperature.
set -euo pipefail

LANE="${1:?usage: run_graphene_fd_thermal_labels_wave2.sh A|B}"
case "$LANE" in
  A)
    SNAPSHOTS=/data/gr_dftmd_recovery/A/T300/md_checkpoint/snapshots.npz
    DEGAUSS=0.0019000869
    LATTICE_TEMPERATURE=300
    ;;
  B)
    SNAPSHOTS=/data/gr_dftmd_recovery/B/T600/md_checkpoint/snapshots.npz
    DEGAUSS=0.0038001738
    LATTICE_TEMPERATURE=600
    ;;
  *) echo "unknown lane '$LANE'" >&2; exit 2 ;;
esac

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
WAVE1="/data/graphene_fd_thermal_labels/$LANE"
WORK="${WORK:-/data/graphene_fd_thermal_labels_wave2/$LANE}"
INDICES="2,6,10,15,19,23,27,31,36,40,44,48,52,54,57"
VALIDATION_INDICES="2,31,57"
KGRID="${KGRID:-8}"

while [[ ! -e "$WAVE1/DONE" ]]; do sleep 60; done
mkdir -p "$WORK/labels"
exec 9>"$WORK/.lock"
if ! flock -n 9; then
    echo "wave 2 lane $LANE is already running; refusing a duplicate"
    exit 0
fi
if [[ -e "$WORK/DONE" ]]; then
    echo "wave 2 lane $LANE already complete"
    exit 0
fi
exec > >(tee -a "$WORK/run.log") 2>&1
rm -f "$WORK/DONE"
export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
echo "=== graphene physical-FD thermal labels wave2 lane=$LANE start $(date -Is) ==="
echo "KGRID=$KGRID INDICES=$INDICES VALIDATION_INDICES=$VALIDATION_INDICES"
"$CONDA" run --no-capture-output -n phonon python \
    "$ROOT/scripts/v100/graphene_fd_force_convergence.py" \
    --snapshots "$SNAPSHOTS" \
    --indices "$INDICES" --validation-indices "$VALIDATION_INDICES" \
    --kgrids "$KGRID" --disk-io none \
    --degauss "$DEGAUSS" --lattice-temperature "$LATTICE_TEMPERATURE" \
    --pw /root/gpupw.sh --pseudo-dir "$ROOT/pseudo" \
    --workdir "$WORK" --output "$WORK/summary.json"
test -s "$WORK/summary.json"
test -s "$WORK/summary.xyz"
touch "$WORK/DONE"
echo "=== graphene physical-FD thermal labels wave2 lane=$LANE COMPLETE $(date -Is) ==="
