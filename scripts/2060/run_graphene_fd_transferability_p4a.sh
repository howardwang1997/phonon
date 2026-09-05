#!/usr/bin/env bash
# Run the frozen 450 K on-policy conditioned-model trajectories and short-TDEP gate.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
RESULTS="$ROOT/results/graphene_fd_transferability"
OUT="$RESULTS/T450/on_policy"
TD="$ROOT/results/td_phonon"
FREEZE="$RESULTS/freeze_manifest.json"
BASE="$ROOT/results/gr_backbone_v11/ft_graphene.model"
DELTA300="$ROOT/results/graphene_fd_delta_pilot/T300/delta32/gr_fd300_delta32.model"
DELTA600="$ROOT/results/graphene_fd_delta_weighted/T600/selected_checkpoint.model"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
OPERATOR="$RESULTS/conditioned_short/T450_long_range_operator.npz"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT/checkpoints" "$TD"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$OUT/DONE" ]]; then
    echo "450 K on-policy sampling stage already complete"
    exit 0
fi
for path in "$FREEZE" "$BASE" "$DELTA300" "$DELTA600" "$BG" "$OPERATOR"; do
    [[ -s "$path" ]]
done
"$CONDA" run -n phonon python -c \
    'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p["status"] == "frozen_before_450_holdout" and not p["direct_450_targets_read"] else 1)' \
    "$FREEZE"

run_seed() {
    local seed="$1"
    local dt="$2"
    local equil="$3"
    local stride="$4"
    local suffix="dt${dt//./p}"
    local checkpoint_root="$OUT/checkpoints/seed${seed}_${suffix}"
    local checkpoint="$checkpoint_root/T450"
    local short_tdep="$OUT/short_seed${seed}.npz"
    if [[ -s "$checkpoint/snapshots.npz" && -s "$short_tdep" ]]; then
        local count
        count="$($CONDA run -n phonon python -c 'import numpy as np,sys; print(len(np.load(sys.argv[1],allow_pickle=False)["positions"]))' "$checkpoint/snapshots.npz" | tail -1)"
        if [[ "$count" == 120 ]]; then
            echo "reuse complete T450 seed=$seed checkpoint"
            printf '%s\n' "$checkpoint" > "$OUT/seed${seed}_checkpoint_path.txt"
            return 0
        fi
    fi
    echo "=== T450 conditioned sampling seed=$seed dt=$dt START $(date -Is) ==="
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/td_phonon_friedel.py \
        --model "$BASE" --delta-model-300 "$DELTA300" --delta-model-600 "$DELTA600" \
        --bg "$BG" --operator "$OPERATOR" --friedel-tel 450 --device cuda \
        --tag "graphene_fd_transferability_conditioned_T450_full_seed${seed}" \
        --temperatures 450 --dt "$dt" --equil "$equil" --nsnap 120 --stride "$stride" \
        --npoints 201 --seed "$seed" --checkpoint-root "$checkpoint_root" \
        --checkpoint-every 25 --max-temperature-factor 5 \
        --min-pair-distance 0.8 --max-force 100
    printf '%s\n' "$checkpoint" > "$OUT/seed${seed}_checkpoint_path.txt"
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
        --background "$BG" --checkpoint "$checkpoint" --subtract-operator "$OPERATOR" \
        --temperature 450 --tag "graphene_fd_transferability_conditioned_T450_short_seed${seed}" \
        --output "$short_tdep"
}

echo "=== graphene transferability P4a START $(date -Is) ==="
run_seed 0 0.5 3000 80
run_seed 1 0.25 6000 160
run_seed 2 0.5 3000 80

checkpoints=()
for seed in 0 1 2; do
    checkpoints+=("$(<"$OUT/seed${seed}_checkpoint_path.txt")")
done
POOLED="$OUT/short_pooled360.npz"
if [[ ! -s "$POOLED" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
        --background "$BG" \
        --checkpoint "${checkpoints[0]}" --checkpoint "${checkpoints[1]}" \
        --checkpoint "${checkpoints[2]}" --subtract-operator "$OPERATOR" \
        --temperature 450 --tag graphene_fd_transferability_conditioned_T450_short_pooled360 \
        --output "$POOLED"
fi

if "$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/summarize_graphene_fd_conditioned_tdep.py \
    --temperature 450 --freeze-manifest "$FREEZE" --operator "$OPERATOR" \
    --checkpoint "${checkpoints[0]}" --checkpoint "${checkpoints[1]}" \
    --checkpoint "${checkpoints[2]}" \
    --seed-tdep "$OUT/short_seed0.npz" --seed-tdep "$OUT/short_seed1.npz" \
    --seed-tdep "$OUT/short_seed2.npz" --pooled-tdep "$POOLED" \
    --seed-mae-threshold 5 --mean-temperature-relative-threshold 0.05 \
    --output "$OUT/sampling_acceptance.json"; then
    touch "$OUT/PASSED_SAMPLING_GATE" "$OUT/READY_DFT_FORCE_LABELS"
    echo "450 K sampling gate passed; fixed DFT force-label shards may be released"
else
    touch "$OUT/BLOCKED_SAMPLING_GATE" "$OUT/DONE"
    echo "450 K sampling gate failed; DFT force labels remain locked"
    exit 0
fi
touch "$OUT/DONE"
echo "=== graphene transferability P4a COMPLETE $(date -Is) ==="
