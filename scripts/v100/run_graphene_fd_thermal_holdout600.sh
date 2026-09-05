#!/usr/bin/env bash
# Fixed, previously unused 600 K physical-FD force holdout (15/15 snapshots).
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
SNAPSHOTS="${SNAPSHOTS:-/data/graphene_fd_thermal_holdout600/snapshots.npz}"
WORK="${WORK:-/data/graphene_fd_thermal_holdout600}"
INDICES="3,7,11,14,18,22,26,30,32,35,39,43,47,51,56"
REMOTE="howardwang@100.105.21.7"
DEST="/home/howardwang/phonon/data/graphene_fd_thermal_holdout600"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
SCP=(scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)

mkdir -p "$WORK/labels"
exec 9>"$WORK/.lock"
if ! flock -n 9; then
    echo "600 K holdout is already running; refusing a duplicate"
    exit 0
fi
if [[ -e "$WORK/DONE" ]]; then
    exit 0
fi
exec > >(tee -a "$WORK/run.log") 2>&1
export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"

for path in "$SNAPSHOTS" "$ROOT/pseudo/C_ONCV_PBE-1.2.upf"; do
    [[ -s "$path" ]]
done
echo "=== fixed 600 K physical-FD force holdout start $(date -Is) ==="
echo "INDICES=$INDICES; all points remain outside training and checkpoint selection"
"$CONDA" run --no-capture-output -n phonon python \
    "$ROOT/scripts/v100/graphene_fd_force_convergence.py" \
    --snapshots "$SNAPSHOTS" --indices "$INDICES" \
    --validation-indices "$INDICES" --kgrids 8 --disk-io none \
    --degauss 0.0038001738 --lattice-temperature 600 \
    --pw /root/gpupw.sh --pseudo-dir "$ROOT/pseudo" \
    --workdir "$WORK" --output "$WORK/summary.json"
test -s "$WORK/summary.json"
test -s "$WORK/summary.xyz"
touch "$WORK/DONE"

"${SSH[@]}" "$REMOTE" "mkdir -p '$DEST'"
for source in "$WORK/summary.json" "$WORK/summary.xyz" "$WORK/run.log"; do
    filename="$(basename "$source")"
    "${SCP[@]}" "$source" "$REMOTE:$DEST/$filename.partial"
    "${SSH[@]}" "$REMOTE" "mv '$DEST/$filename.partial' '$DEST/$filename'"
done
"${SSH[@]}" "$REMOTE" "touch '$DEST/RAW_READY'"
touch "$WORK/RELAYED_TO_2060"
echo "=== fixed 600 K physical-FD force holdout COMPLETE $(date -Is) ==="
