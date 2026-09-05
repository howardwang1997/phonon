#!/usr/bin/env bash
# Run one six-configuration R1 DFT overlap shard after freeze-manifest checks.
set -euo pipefail

SHARD="${1:?usage: run_graphene_fixed_smearing_overlap_shard.sh A|B}"
case "$SHARD" in A|B) ;; *) echo "SHARD must be A or B" >&2; exit 2 ;; esac

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
WORK_ROOT="${WORK_ROOT:-/data/graphene_fixed_smearing_overlap_R1}"
INPUT_ROOT="$WORK_ROOT/input"
WORK="$WORK_ROOT/shard_$SHARD"
MANIFEST="$INPUT_ROOT/freeze_manifest.json"
SNAPSHOTS="$INPUT_ROOT/shard_${SHARD}_snapshots.npz"
LABEL_SCRIPT="$ROOT/scripts/v100/graphene_fixed_smearing_overlap_labels.py"

mkdir -p "$WORK"
exec 9>"$WORK/.lock"
if ! flock -n 9; then
    echo "R1 shard $SHARD is already running"
    exit 0
fi
if [[ -e "$WORK/DONE" ]]; then
    echo "R1 shard $SHARD is already complete"
    exit 0
fi
exec > >(tee -a "$WORK/run.log") 2>&1

for path in "$MANIFEST" "$SNAPSHOTS" "$LABEL_SCRIPT" \
    "$ROOT/pseudo/C_ONCV_PBE-1.2.upf" /root/gpupw.sh; do
    test -s "$path"
done

"$CONDA" run -n phonon python -c '
import hashlib, json, pathlib, sys
manifest_path, snapshots_path, shard = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
payload = json.loads(manifest_path.read_text())
if payload.get("status") != "frozen_before_R1_DFT":
    raise SystemExit("R1 manifest is not frozen")
if payload["condition"] != {
    "lattice_temperature_K": 450.0,
    "smearing": "fermi-dirac",
    "degauss_Ry": 0.0019000869380739254,
    "cell": "fixed operator/SSCHA cell",
    "n_atoms": 72,
}:
    raise SystemExit("R1 condition differs from frozen protocol")
expected = payload["outputs"][f"shard_{shard}_snapshots_sha256"]
observed = hashlib.sha256(snapshots_path.read_bytes()).hexdigest()
if observed != expected:
    raise SystemExit(f"snapshot hash mismatch: {observed} != {expected}")
if len(payload["shards"][shard]) != 6:
    raise SystemExit("frozen shard does not contain six structures")
print(f"validated R1 shard {shard}: {observed}")
' "$MANIFEST" "$SNAPSHOTS" "$SHARD"

while pgrep -x pw.x >/dev/null || pgrep -x ph.x >/dev/null; do
    echo "waiting for active QE calculation: $(date -Is)"
    sleep 60
done

date -Is > "$WORK/STARTED_AT"
touch "$WORK/RUNNING"
finish() {
    local exit_code=$?
    rm -f "$WORK/RUNNING"
    echo "$exit_code" > "$WORK/EXIT_CODE"
    if (( exit_code != 0 )); then
        touch "$WORK/FAILED"
    fi
}
trap finish EXIT

export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
echo "=== R1 fixed-smearing overlap shard=$SHARD START $(date -Is) ==="
"$CONDA" run --no-capture-output -n phonon python "$LABEL_SCRIPT" \
    --snapshots "$SNAPSHOTS" --indices 0,1,2,3,4,5 \
    --degauss 0.0019000869 --lattice-temperature 450 --kgrid 8 \
    --pw /root/gpupw.sh --pseudo-dir "$ROOT/pseudo" \
    --workdir "$WORK" --output "$WORK/summary.json"
test -s "$WORK/summary.json"
test -s "$WORK/summary.xyz"
date -Is > "$WORK/COMPLETED_AT"
touch "$WORK/DONE"
echo "=== R1 fixed-smearing overlap shard=$SHARD COMPLETE $(date -Is) ==="
