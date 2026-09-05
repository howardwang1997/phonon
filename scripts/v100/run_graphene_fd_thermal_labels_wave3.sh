#!/usr/bin/env bash
# Conditional third label wave.  It runs only when the expanded wave-2 model
# fails the fixed force/TDEP/seed thresholds recorded in acceptance.json.
set -euo pipefail

LANE="${1:?usage: run_graphene_fd_thermal_labels_wave3.sh A|B}"
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
REMOTE="howardwang@100.105.21.7"
ACCEPTANCE=/home/howardwang/phonon/results/graphene_fd_thermal_finetune_wave2/acceptance.json
WAVE2="/data/graphene_fd_thermal_labels_wave2/$LANE"
WORK="${WORK:-/data/graphene_fd_thermal_labels_wave3/$LANE}"
INDICES="1,5,9,12,16,20,24,28,33,37,41,45,49,53,58"
VALIDATION_INDICES="1,33,58"
KGRID="${KGRID:-8}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)

while ! "${SSH[@]}" "$REMOTE" "test -s '$ACCEPTANCE'"; do sleep 300; done
required="$("${SSH[@]}" "$REMOTE" \
    "/home/howardwang/miniconda3/bin/conda run -n phonon python -c \
    'import json; print(int(json.load(open(\"$ACCEPTANCE\"))[\"wave3_required\"]))'" | tail -1)"
if [[ "$required" != 1 ]]; then
    mkdir -p "$WORK"
    touch "$WORK/SKIPPED_GATE_PASSED"
    echo "wave 2 passed; wave 3 lane $LANE skipped"
    exit 0
fi
while [[ ! -e "$WAVE2/DONE" ]]; do sleep 60; done

mkdir -p "$WORK/labels"
exec 9>"$WORK/.lock"
if ! flock -n 9; then exit 0; fi
if [[ -e "$WORK/DONE" ]]; then exit 0; fi
exec > >(tee -a "$WORK/run.log") 2>&1
rm -f "$WORK/DONE"
export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
echo "=== graphene physical-FD thermal labels wave3 lane=$LANE start $(date -Is) ==="
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
echo "=== graphene physical-FD thermal labels wave3 lane=$LANE COMPLETE $(date -Is) ==="
