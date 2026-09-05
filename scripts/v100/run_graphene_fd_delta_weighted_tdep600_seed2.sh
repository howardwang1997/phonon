#!/usr/bin/env bash
# Parallel third-seed lane; its artifacts are copied back to the V100-A aggregate lane.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
MLIP_ENV="${MLIP_ENV:-phonon-mlip}"
TD="$ROOT/results/td_phonon"
OUT="$ROOT/results/graphene_fd_delta_weighted/T600_TDEP_seed2"
MODEL="$ROOT/results/graphene_fd_delta_weighted/T600/selected_checkpoint.model"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
OPERATOR="$ROOT/data/graphene_fd_delta_pilot/T600/long_range_operator.npz"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
TAG=graphene_v11_fd600_delta_weighted_full_sampling_seed2

cd "$ROOT"
mkdir -p "$OUT" "$TD"
exec > >(tee -a "$OUT/run.log") 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$OUT/DONE" ]]; then
    exit 0
fi
for path in "$MODEL" "$V11" "$OPERATOR" "$BG"; do
    test -s "$path"
done
echo "=== weighted-delta 600 K seed 2 START $(date -Is); mlip_env=$MLIP_ENV ==="
sha256sum "$MODEL" "$V11" "$OPERATOR" "$BG"
"$CONDA" run --no-capture-output -n "$MLIP_ENV" python \
    scripts/smearing_kink/td_phonon_friedel.py \
    --model "$V11" --delta-model "$MODEL" --bg "$BG" \
    --operator "$OPERATOR" --device cuda --tag "$TAG" \
    --temperatures 600 --dt 0.5 --equil 3000 \
    --nsnap 120 --stride 80 --npoints 201 --seed 2 \
    --checkpoint-root "results/td_phonon/${TAG}_checkpoint" \
    --checkpoint-every 25 --max-temperature-factor 5 \
    --min-pair-distance 0.8 --max-force 100
test -s "$TD/td_${TAG}.npz"
test -s "$TD/td_${TAG}.csv"
touch "$OUT/DONE"
echo "=== weighted-delta 600 K seed 2 COMPLETE $(date -Is) ==="
