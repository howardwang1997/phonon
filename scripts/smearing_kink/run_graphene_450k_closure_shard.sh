#!/usr/bin/env bash
# Run one 450 K closure DFT shard (6 configs, 8x8x1 k, fd degauss 0.0019 Ry).
# Usage: run_graphene_450k_closure_shard.sh <A|B>
set -uo pipefail

SHARD="${1:?usage: $0 <A|B>}"
ROOT=/data/graphene_450k_closure
WORK="$ROOT/T450/shard_$SHARD"
SNAPSHOT="$ROOT/input/shard_${SHARD}_snapshots.npz"
LABEL_SCRIPT="$ROOT/code/graphene_fd_force_convergence.py"

mkdir -p "$WORK"
cd "$WORK"
if [ -f DONE ]; then
    echo "shard $SHARD already DONE"
    exit 0
fi
exec 9>"$WORK/.lock"
flock -n 9 || { echo "another instance holds the lock"; exit 1; }

date -Is > STARTED_AT
if /root/miniconda3/bin/conda run -n phonon python "$LABEL_SCRIPT" \
    --snapshots "$SNAPSHOT" \
    --indices 0,1,2,3,4,5 \
    --kgrids 8 \
    --degauss 0.0019000869380739254 \
    --lattice-temperature 450 \
    --disk-io none \
    --pw /root/gpupw.sh \
    --pseudo-dir /root/phonon/pseudo \
    --workdir "$WORK" \
    --output "$WORK/summary.json" > run.log 2>&1; then
    CODE=0
else
    CODE=$?
fi
echo "$CODE" > EXIT_CODE
date -Is > COMPLETED_AT
if [ "$CODE" -eq 0 ]; then
    touch DONE
fi
exit "$CODE"
