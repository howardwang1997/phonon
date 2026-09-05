#!/usr/bin/env bash
# Merge the two disjoint 600 K force-holdout shards on the 2060 coordinator.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
OUT="$ROOT/data/graphene_fd_thermal_holdout600"
LOG="$OUT/merge.log"

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
while [[ ! -e "$OUT/shard_A/RAW_READY" || ! -e "$OUT/shard_B/RAW_READY" ]]; do
    echo "waiting for both disjoint 600 K holdout shards: $(date -Is)"
    sleep 60
done
if [[ -e "$OUT/RAW_READY" && -s "$OUT/summary.json" && -s "$OUT/summary.xyz" ]]; then
    exit 0
fi
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/merge_graphene_fd_holdout_shards.py \
    --shard "A=$OUT/shard_A/summary.json" \
    --shard "B=$OUT/shard_B/summary.json" \
    --expected-indices 3,7,11,14,18,22,26,30,32,35,39,43,47,51,56 \
    --output-json "$OUT/summary.json" --output-xyz "$OUT/summary.xyz"
touch "$OUT/RAW_READY"
echo "=== fixed 600 K holdout shards merged $(date -Is) ==="
