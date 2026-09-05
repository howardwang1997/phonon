#!/usr/bin/env bash
# Same-backbone control for the finite-lattice-temperature graphene comparison.
# It deliberately omits the Friedel/q-space term while keeping the v11 model,
# DFT lattice constant, supercell, temperatures, and TDEP sampling explicit.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
MODEL="${MODEL:-results/gr_backbone_v11/ft_graphene.model}"
TAG="graphene_v11_no_long_range_physical_geom"
OUTDIR="$ROOT/results/td_phonon"
CHECKPOINT_ROOT="$OUTDIR/${TAG}_checkpoint"
DONE="$OUTDIR/${TAG}.DONE"
LOG="$OUTDIR/${TAG}.log"

cd "$ROOT"
mkdir -p "$OUTDIR" "$CHECKPOINT_ROOT"
if [[ -s "$OUTDIR/td_${TAG}.npz" && -s "$OUTDIR/td_${TAG}.csv" ]]; then
    if "$CONDA" run -n phonon python -c '
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as data:
    assert data["temperatures"].tolist() == [300.0, 600.0]
    assert all(key in data for key in ("T300_freq", "T600_freq"))
' "$OUTDIR/td_${TAG}.npz"; then
        touch "$DONE"
        echo "same-backbone control already complete"
        exit 0
    fi
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== graphene v11 no-long-range control start $(date -Is) ==="
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
"$CONDA" run --no-capture-output -n phonon python scripts/td_phonon.py \
    --structure data/td_phonon/graphene.xyz \
    --model "$MODEL" --device cuda \
    --tag "$TAG" --outdir results/td_phonon \
    --temperatures 300,600 --supercell 6,6,1 \
    --no-relax --a 2.4600000087 \
    --equil 1500 --nsnap 120 --stride 40 --npoints 201 \
    --checkpoint-root "results/td_phonon/${TAG}_checkpoint" \
    --checkpoint-every 25
"$CONDA" run -n phonon python -c '
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as data:
    assert data["temperatures"].tolist() == [300.0, 600.0]
    assert all(key in data for key in ("T300_freq", "T600_freq"))
' "$OUTDIR/td_${TAG}.npz"
touch "$DONE"
echo "=== graphene v11 no-long-range control COMPLETE $(date -Is) ==="
