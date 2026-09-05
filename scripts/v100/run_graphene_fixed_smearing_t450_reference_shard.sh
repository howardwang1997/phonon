#!/usr/bin/env bash
# Relabel the frozen 450 K on-policy configurations at the same electronic
# smearing used by the E48 300/450 K comparison.  This campaign writes to a
# new directory and never modifies the original 0.00285013035 Ry P4 labels.
set -euo pipefail

SHARD="${1:?usage: run_graphene_fixed_smearing_t450_reference_shard.sh A|B}"
ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
LABEL_SCRIPT="${LABEL_SCRIPT:-$ROOT/scripts/v100/graphene_fd_force_convergence.py}"
SOURCE_ROOT="/data/graphene_fd_transferability_p4b"
WORK_ROOT="${WORK_ROOT:-/data/graphene_fixed_smearing_thermal_reference/T450}"
WORK="$WORK_ROOT/shard_${SHARD}"
TARGET_DEGAUSS_RY="0.0019000869"
ALL_INDICES="3,9,15,21,27,33,39,45,51,57,63,69,75,81,87,93,99,105,111,117"
FIRST_INDICES="3,9,15,21,27,33,39,45,51,57"
LAST_INDICES="63,69,75,81,87,93,99,105,111,117"

case "$SHARD" in
  A)
    SOURCE_SHARD="A"
    LANE_SPECS=("0:$ALL_INDICES" "2:$FIRST_INDICES")
    EXPECTED_HASHES=(
      "0:e85f26f01c20a0d882609a543cff36a2eaab994826a37371b6177f02fae7554a"
      "2:534ada6895a99126a035f5bf46d03033640073ec03f9102a7f6fbfb7bbef3b5c"
    )
    ;;
  B)
    SOURCE_SHARD="B"
    LANE_SPECS=("1:$ALL_INDICES" "2:$LAST_INDICES")
    EXPECTED_HASHES=(
      "1:df502481e400e142a86bb732ff595cbe75381e19fc8a4a6be75d8c63b0fcea7e"
      "2:534ada6895a99126a035f5bf46d03033640073ec03f9102a7f6fbfb7bbef3b5c"
    )
    ;;
  *)
    echo "SHARD must be A or B" >&2
    exit 2
    ;;
esac

mkdir -p "$WORK"
exec 9>"$WORK/.lock"
if ! flock -n 9; then
  echo "fixed-smearing T450 shard $SHARD is already running"
  exit 0
fi
if [[ -e "$WORK/DONE" ]]; then
  echo "fixed-smearing T450 shard $SHARD is already complete"
  exit 0
fi
exec > >(tee -a "$WORK/run.log") 2>&1

while pgrep -x pw.x >/dev/null || pgrep -x ph.x >/dev/null; do
  echo "waiting for the active QE calculation: $(date -Is)"
  sleep 60
done

test -x /root/gpupw.sh
test -f "$ROOT/pseudo/C_ONCV_PBE-1.2.upf"
test -f "$LABEL_SCRIPT"

for item in "${EXPECTED_HASHES[@]}"; do
  seed="${item%%:*}"
  expected="${item#*:}"
  snapshots="$SOURCE_ROOT/shard_${SOURCE_SHARD}/inputs/seed${seed}_snapshots.npz"
  test -s "$snapshots"
  observed="$(sha256sum "$snapshots" | awk '{print $1}')"
  if [[ "$observed" != "$expected" ]]; then
    echo "seed${seed} snapshot hash mismatch: $observed != $expected" >&2
    exit 1
  fi
done

echo "=== fixed-smearing T450 DFT reference shard=$SHARD start $(date -Is) ==="
export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
for specification in "${LANE_SPECS[@]}"; do
  seed="${specification%%:*}"
  indices="${specification#*:}"
  snapshots="$SOURCE_ROOT/shard_${SOURCE_SHARD}/inputs/seed${seed}_snapshots.npz"
  lane="$WORK/seed${seed}"
  mkdir -p "$lane/labels"
  "$CONDA" run --no-capture-output -n phonon python \
    "$LABEL_SCRIPT" \
    --snapshots "$snapshots" \
    --indices "$indices" --validation-indices "$indices" \
    --trajectory-seed "$seed" --kgrids 8 --disk-io none \
    --degauss "$TARGET_DEGAUSS_RY" --lattice-temperature 450 \
    --pw /root/gpupw.sh --pseudo-dir "$ROOT/pseudo" \
    --workdir "$lane" --output "$lane/summary.json"
  test -s "$lane/summary.json"
  test -s "$lane/summary.xyz"
done

"$CONDA" run --no-capture-output -n phonon python -c \
  'import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = float(sys.argv[2])
summaries = sorted(root.glob("seed*/summary.json"))
if len(summaries) != 2:
    raise SystemExit(f"expected two lane summaries, found {len(summaries)}")
count = 0
for path in summaries:
    payload = json.loads(path.read_text())
    if payload["smearing"] != "fermi-dirac":
        raise SystemExit(f"wrong smearing in {path}")
    if abs(float(payload["degauss_Ry"]) - expected) > 5e-11:
        raise SystemExit(f"wrong degauss in {path}")
    if float(payload["lattice_temperature_K"]) != 450.0:
        raise SystemExit(f"wrong lattice temperature in {path}")
    if payload["kgrids"] != [8] or int(payload["n_atoms"]) != 72:
        raise SystemExit(f"wrong cell or k grid in {path}")
    count += len(payload["indices"])
if count != 30:
    raise SystemExit(f"expected 30 labels, found {count}")
print(f"validated {count} labels in {root}")' "$WORK" "$TARGET_DEGAUSS_RY"

touch "$WORK/DONE"
echo "=== fixed-smearing T450 DFT reference shard=$SHARD COMPLETE $(date -Is) ==="
