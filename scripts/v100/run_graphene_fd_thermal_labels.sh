#!/usr/bin/env bash
# Restartable physical-FD labels for the finite-temperature graphene MLIP.
set -euo pipefail

LANE="${1:?usage: bash scripts/v100/run_graphene_fd_thermal_labels.sh A|B}"
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
WORK="${WORK:-/data/graphene_fd_thermal_labels/$LANE}"
INDICES="${INDICES:-0,4,8,13,17,21,25,29,34,38,42,46,50,55,59}"
VALIDATION_INDICES="${VALIDATION_INDICES:-0,29,59}"
KGRID="${KGRID:-8}"
mkdir -p "$WORK/labels"
exec 9>"$WORK/.lock"
if ! flock -n 9; then
  echo "lane $LANE is already running; refusing a duplicate" >&2
  exit 0
fi
exec > >(tee -a "$WORK/run.log") 2>&1
rm -f "$WORK/DONE"

# Reuse the three high-density validation geometries from the convergence run.
for index in 0 29 59; do
  source_label="/data/graphene_fd_force_convergence/$LANE/labels/snapshot_$(printf '%03d' "$index")_k${KGRID}.npz"
  if [[ -s "$source_label" ]]; then
    cp -p "$source_label" "$WORK/labels/"
  fi
done

export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
echo "=== graphene physical-FD thermal labels lane=$LANE start $(date -Is) ==="
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
touch "$WORK/DONE"
echo "=== graphene physical-FD thermal labels lane=$LANE COMPLETE $(date -Is) ==="
