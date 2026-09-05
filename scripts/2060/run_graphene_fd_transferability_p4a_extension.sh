#!/usr/bin/env bash
# Extend the three frozen 450 K MLIP trajectories after bootstrap insufficiency.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
RESULTS="$ROOT/results/graphene_fd_transferability"
OUT="$RESULTS/T450/on_policy"
FREEZE="$RESULTS/freeze_manifest.json"
BASE="$ROOT/results/gr_backbone_v11/ft_graphene.model"
DELTA300="$ROOT/results/graphene_fd_delta_pilot/T300/delta32/gr_fd300_delta32.model"
DELTA600="$ROOT/results/graphene_fd_delta_weighted/T600/selected_checkpoint.model"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
OPERATOR="$RESULTS/conditioned_short/T450_long_range_operator.npz"
TARGET=3000
BOOTSTRAP_N120="$OUT/sampling_bootstrap_acceptance_n120.json"
LOG="$OUT/run_extension_n${TARGET}.log"

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$OUT/EXTENSION_N${TARGET}_DONE" ]]; then
    echo "450 K n=$TARGET sampling extension already complete"
    exit 0
fi
if pgrep -f '[t]d_phonon_friedel.py' >/dev/null; then
    echo "another MLIP-MD sampling process is active; retrying later" >&2
    exit 75
fi

for path in "$FREEZE" "$BASE" "$DELTA300" "$DELTA600" "$BG" "$OPERATOR" \
    "$OUT/sampling_acceptance.json" "$OUT/short_seed0.npz" \
    "$OUT/short_seed1.npz" "$OUT/short_seed2.npz"; do
    [[ -s "$path" ]]
done

checkpoints=(
    "$OUT/checkpoints/seed0_dt0p5/T450"
    "$OUT/checkpoints/seed1_dt0p25/T450"
    "$OUT/checkpoints/seed2_dt0p5/T450"
)

if [[ ! -s "$BOOTSTRAP_N120" ]]; then
    set +e
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/bootstrap_graphene_fd_conditioned_sampling.py \
        --background "$BG" --operator "$OPERATOR" \
        --checkpoint "0=${checkpoints[0]}/snapshots.npz" \
        --checkpoint "1=${checkpoints[1]}/snapshots.npz" \
        --checkpoint "2=${checkpoints[2]}/snapshots.npz" \
        --seed-tdep "0=$OUT/short_seed0.npz" \
        --seed-tdep "1=$OUT/short_seed1.npz" \
        --seed-tdep "2=$OUT/short_seed2.npz" \
        --freeze-manifest "$FREEZE" --sampling-acceptance "$OUT/sampling_acceptance.json" \
        --temperature 450 --replicates 1000 --checkpoint-every 20 \
        --output "$BOOTSTRAP_N120"
    bootstrap_status=$?
    set -e
    if [[ "$bootstrap_status" -ne 1 ]]; then
        echo "unexpected initial bootstrap exit status: $bootstrap_status" >&2
        exit 2
    fi
fi
"$CONDA" run -n phonon python -c \
    'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p["status"] == "insufficient_sampling" and not p["passes_sampling_bootstrap_gate"] else 1)' \
    "$BOOTSTRAP_N120"

if [[ -e "$OUT/READY_DFT_FORCE_LABELS" ]]; then
    mv "$OUT/READY_DFT_FORCE_LABELS" "$OUT/READY_DFT_FORCE_LABELS_POINT_ONLY_REVOKED"
fi
touch "$OUT/BLOCKED_SAMPLING_BOOTSTRAP_N120"

run_seed() {
    local seed="$1"
    local dt="$2"
    local equil="$3"
    local stride="$4"
    local suffix="dt${dt//./p}"
    local checkpoint_root="$OUT/checkpoints/seed${seed}_${suffix}"
    local checkpoint="$checkpoint_root/T450"
    local short_tdep="$OUT/short_seed${seed}_n${TARGET}.npz"
    local count
    count="$($CONDA run -n phonon python -c 'import numpy as np,sys; print(len(np.load(sys.argv[1],allow_pickle=False)["positions"]))' "$checkpoint/snapshots.npz" | tail -1)"
    if [[ "$count" -lt "$TARGET" ]]; then
        echo "=== extend T450 seed=$seed snapshots=$count->$TARGET START $(date -Is) ==="
        "$CONDA" run --no-capture-output -n phonon python \
            scripts/smearing_kink/td_phonon_friedel.py \
            --model "$BASE" --delta-model-300 "$DELTA300" --delta-model-600 "$DELTA600" \
            --bg "$BG" --operator "$OPERATOR" --friedel-tel 450 --device cuda \
            --tag "graphene_fd_transferability_conditioned_T450_extension_n${TARGET}_seed${seed}" \
            --temperatures 450 --dt "$dt" --equil "$equil" --nsnap "$TARGET" --stride "$stride" \
            --npoints 201 --seed "$seed" --checkpoint-root "$checkpoint_root" \
            --checkpoint-every 25 --max-temperature-factor 5 \
            --min-pair-distance 0.8 --max-force 100
    fi
    count="$($CONDA run -n phonon python -c 'import numpy as np,sys; print(len(np.load(sys.argv[1],allow_pickle=False)["positions"]))' "$checkpoint/snapshots.npz" | tail -1)"
    [[ "$count" == "$TARGET" ]]
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
        --background "$BG" --checkpoint "$checkpoint" --subtract-operator "$OPERATOR" \
        --temperature 450 --tag "graphene_fd_transferability_conditioned_T450_short_seed${seed}_n${TARGET}" \
        --output "$short_tdep"
}

echo "=== graphene transferability P4a extension START $(date -Is) ==="
run_seed 0 0.5 3000 80
run_seed 1 0.25 6000 160
run_seed 2 0.5 3000 80

POOLED="$OUT/short_pooled_n$((TARGET * 3)).npz"
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
    --background "$BG" \
    --checkpoint "${checkpoints[0]}" --checkpoint "${checkpoints[1]}" \
    --checkpoint "${checkpoints[2]}" --subtract-operator "$OPERATOR" \
    --temperature 450 --tag "graphene_fd_transferability_conditioned_T450_short_pooled_n$((TARGET * 3))" \
    --output "$POOLED"

POINT="$OUT/sampling_acceptance_n${TARGET}.json"
if ! "$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/summarize_graphene_fd_conditioned_tdep.py \
    --temperature 450 --freeze-manifest "$FREEZE" --operator "$OPERATOR" \
    --checkpoint "${checkpoints[0]}" --checkpoint "${checkpoints[1]}" \
    --checkpoint "${checkpoints[2]}" \
    --seed-tdep "$OUT/short_seed0_n${TARGET}.npz" \
    --seed-tdep "$OUT/short_seed1_n${TARGET}.npz" \
    --seed-tdep "$OUT/short_seed2_n${TARGET}.npz" --pooled-tdep "$POOLED" \
    --seed-mae-threshold 5 --mean-temperature-relative-threshold 0.05 \
    --expected-snapshots-per-seed "$TARGET" --output "$POINT"; then
    touch "$OUT/BLOCKED_SAMPLING_POINT_N${TARGET}" "$OUT/EXTENSION_N${TARGET}_DONE"
    echo "450 K n=$TARGET point gate failed; DFT labels remain locked"
    exit 0
fi

BOOTSTRAP="$OUT/sampling_bootstrap_acceptance_n${TARGET}.json"
if "$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/bootstrap_graphene_fd_conditioned_sampling.py \
    --background "$BG" --operator "$OPERATOR" \
    --checkpoint "0=${checkpoints[0]}/snapshots.npz" \
    --checkpoint "1=${checkpoints[1]}/snapshots.npz" \
    --checkpoint "2=${checkpoints[2]}/snapshots.npz" \
    --seed-tdep "0=$OUT/short_seed0_n${TARGET}.npz" \
    --seed-tdep "1=$OUT/short_seed1_n${TARGET}.npz" \
    --seed-tdep "2=$OUT/short_seed2_n${TARGET}.npz" \
    --freeze-manifest "$FREEZE" --sampling-acceptance "$POINT" \
    --temperature 450 --replicates 1000 --checkpoint-every 20 --output "$BOOTSTRAP"; then
    touch "$OUT/PASSED_SAMPLING_BOOTSTRAP_N${TARGET}" "$OUT/READY_DFT_FORCE_LABELS"
    echo "450 K n=$TARGET point and bootstrap gates passed; DFT labels released"
else
    touch "$OUT/BLOCKED_SAMPLING_BOOTSTRAP_N${TARGET}"
    echo "450 K n=$TARGET bootstrap gate failed; DFT labels remain locked"
fi
touch "$OUT/EXTENSION_N${TARGET}_DONE"
echo "=== graphene transferability P4a extension COMPLETE $(date -Is) ==="
