#!/usr/bin/env bash
# Low-priority filler while the V100 physical-FD labels are being generated.
# Independent seeds quantify the TDEP sampling spread of the same-backbone
# no-long-range control.  Every seed is checkpointed and yields immediately
# when the formal thermal fine-tune data become ready.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
READY="$ROOT/data/graphene_fd_thermal_labels/.READY"
MODEL="results/gr_backbone_v11/ft_graphene.model"
OUTDIR="$ROOT/results/td_phonon"
cd "$ROOT"
mkdir -p "$OUTDIR"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"

for seed in 1 2 3 4 5 6 7 8; do
    [[ -e "$READY" ]] && exit 0
    tag="graphene_v11_no_long_range_seed${seed}"
    if [[ -s "$OUTDIR/td_${tag}.npz" && -s "$OUTDIR/td_${tag}.csv" ]]; then
        echo "[$(date -Is)] seed $seed already complete"
        continue
    fi
    echo "[$(date -Is)] start v11 no-long-range TDEP seed=$seed"
    "$CONDA" run --no-capture-output -n phonon python scripts/td_phonon.py \
        --structure data/td_phonon/graphene.xyz --model "$MODEL" --device cuda \
        --tag "$tag" --outdir results/td_phonon --temperatures 300,600 \
        --supercell 6,6,1 --no-relax --a 2.4600000087 \
        --equil 1500 --nsnap 120 --stride 40 --npoints 201 --seed "$seed" \
        --checkpoint-root "results/td_phonon/${tag}_checkpoint" \
        --checkpoint-every 25
done
echo "[$(date -Is)] v11 no-long-range independent-seed sweep complete"
