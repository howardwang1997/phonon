#!/usr/bin/env bash
# Run one 450 K closure DFT batch (any size snapshot npz, 8x8x1 k, fd degauss 0.0019 Ry).
# Usage: run_graphene_450k_closure_batch.sh <snapshot.npz> <workdir-name>
set -uo pipefail

SNAPSHOT="${1:?usage: $0 <snapshot.npz> <name>}"
NAME="${2:?usage: $0 <snapshot.npz> <name>}"
ROOT=/data/graphene_450k_closure
WORK="$ROOT/T450/$NAME"
LABEL_SCRIPT="$ROOT/code/graphene_fd_force_convergence.py"

N=$(/root/miniconda3/bin/conda run -n phonon python -c "import numpy as np; print(len(np.load('${SNAPSHOT}')['positions']))")
INDICES=$(seq -s, 0 $((N - 1)))
echo "batch $NAME: $N configs (indices $INDICES)"

mkdir -p "$WORK"
cd "$WORK"
if [ -f DONE ]; then
    echo "batch $NAME already DONE"
    exit 0
fi
exec 9>"$WORK/.lock"
flock -n 9 || { echo "another instance holds the lock"; exit 1; }

date -Is > STARTED_AT
if /root/miniconda3/bin/conda run -n phonon python "$LABEL_SCRIPT" \
    --snapshots "$SNAPSHOT" \
    --indices "$INDICES" \
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
