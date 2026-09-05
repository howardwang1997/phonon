#!/usr/bin/env bash
set -euo pipefail

LANE="${1:?usage: bash scripts/v100/run_graphene_fd_force_convergence.sh A|B}"
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
  *)
    echo "unknown lane '$LANE'" >&2
    exit 2
    ;;
esac

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
WORK="${WORK:-/data/graphene_fd_force_convergence/$LANE}"
# Stage the convergence test.  k=12 is the first mesh denser than the existing
# k=8 labels; k=16 is launched only if the measured k=8 -> 12 force difference
# still exceeds the accuracy target.
KGRIDS="${KGRIDS:-4,6,8,12}"
mkdir -p "$WORK"
exec 9>"$WORK/.lock"
if ! flock -n 9; then
  echo "lane $LANE is already running; refusing a duplicate" >&2
  exit 0
fi
exec > >(tee -a "$WORK/run.log") 2>&1
rm -f "$WORK/DONE"

export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
echo "=== graphene FD force convergence lane=$LANE start $(date -Is) ==="
echo "KGRIDS=$KGRIDS"
"$CONDA" run --no-capture-output -n phonon python \
  "$ROOT/scripts/v100/graphene_fd_force_convergence.py" \
  --snapshots "$SNAPSHOTS" \
  --indices 0,29,59 --kgrids "$KGRIDS" \
  --degauss "$DEGAUSS" --lattice-temperature "$LATTICE_TEMPERATURE" \
  --pw /root/gpupw.sh --pseudo-dir "$ROOT/pseudo" \
  --workdir "$WORK" --output "$WORK/summary.json"
test -s "$WORK/summary.json"
touch "$WORK/DONE"
echo "=== graphene FD force convergence lane=$LANE COMPLETE $(date -Is) ==="
