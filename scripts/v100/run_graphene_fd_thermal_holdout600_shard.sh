#!/usr/bin/env bash
# One non-overlapping shard of the fixed 600 K physical-FD force holdout.
set -euo pipefail

SHARD="${1:?usage: run_graphene_fd_thermal_holdout600_shard.sh SHARD INDICES}"
INDICES="${2:?usage: run_graphene_fd_thermal_holdout600_shard.sh SHARD INDICES}"
ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
SNAPSHOTS="${SNAPSHOTS:-/data/graphene_fd_thermal_holdout600/snapshots.npz}"
WORK="${WORK:-/data/graphene_fd_thermal_holdout600_shard_${SHARD}}"
REMOTE="howardwang@100.105.21.7"
DEST="/home/howardwang/phonon/data/graphene_fd_thermal_holdout600/shard_${SHARD}"
SUMMARY="$WORK/summary_shard_${SHARD}.json"
LOG="$WORK/run_shard_${SHARD}.log"
SSH=(env -u LD_LIBRARY_PATH ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
SCP=(env -u LD_LIBRARY_PATH scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)

mkdir -p "$WORK/labels"
exec 9>"$WORK/.shard_${SHARD}.lock"
if ! flock -n 9; then
    echo "600 K holdout shard $SHARD is already running; refusing a duplicate"
    exit 0
fi
if [[ -e "$WORK/SHARD_${SHARD}_DONE" \
   && -e "$WORK/RELAYED_SHARD_${SHARD}_TO_2060" ]]; then
    exit 0
fi
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -e "$WORK/SHARD_${SHARD}_DONE" ]]; then
    for path in "$SNAPSHOTS" "$ROOT/pseudo/C_ONCV_PBE-1.2.upf"; do [[ -s "$path" ]]; done
    echo "=== fixed 600 K physical-FD force holdout shard=$SHARD start $(date -Is) ==="
    echo "INDICES=$INDICES; all points are validation-only and remain outside model selection"
    "$CONDA" run --no-capture-output -n phonon python \
        "$ROOT/scripts/v100/graphene_fd_force_convergence.py" \
        --snapshots "$SNAPSHOTS" --indices "$INDICES" \
        --validation-indices "$INDICES" --kgrids 8 --disk-io none \
        --degauss 0.0038001738 --lattice-temperature 600 \
        --pw /root/gpupw.sh --pseudo-dir "$ROOT/pseudo" \
        --workdir "$WORK" --output "$SUMMARY"
    touch "$WORK/SHARD_${SHARD}_DONE"
else
    echo "shard $SHARD labels already complete; retrying Tailscale relay only"
fi
test -s "$SUMMARY"
test -s "${SUMMARY%.json}.xyz"

"${SSH[@]}" "$REMOTE" "mkdir -p '$DEST'"
for specification in "$SUMMARY:summary.json" "${SUMMARY%.json}.xyz:summary.xyz" "$LOG:run.log"; do
    source="${specification%%:*}"
    filename="${specification##*:}"
    "${SCP[@]}" "$source" "$REMOTE:$DEST/$filename.partial"
    "${SSH[@]}" "$REMOTE" "mv '$DEST/$filename.partial' '$DEST/$filename'"
done
"${SSH[@]}" "$REMOTE" "touch '$DEST/RAW_READY'"
touch "$WORK/RELAYED_SHARD_${SHARD}_TO_2060"
echo "=== fixed 600 K physical-FD force holdout shard=$SHARD COMPLETE $(date -Is) ==="
