#!/usr/bin/env bash
# Sample the reconstructed full force, subtract its provisional LR exactly, then add direct-DFPT q-space LR.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
PHONON_ENV="${PHONON_ENV:-phonon}"
MLIP_ENV="${MLIP_ENV:-phonon-mlip}"
LOCAL_HOLDOUT="${LOCAL_HOLDOUT:-/data/graphene_fd_thermal_holdout600}"
OUT="$ROOT/results/graphene_fd_delta_weighted/T600_TDEP"
TD="$ROOT/results/td_phonon"
REMOTE="howardwang@100.105.21.7"
REMOTE_ROOT=/home/howardwang/phonon
REMOTE_WEIGHTED="$REMOTE_ROOT/results/graphene_fd_delta_weighted"
LOCAL_WEIGHTED="$ROOT/results/graphene_fd_delta_weighted"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
OPERATOR="$ROOT/data/graphene_fd_delta_pilot/T600/long_range_operator.npz"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
DFT="$TD/graphene_physical_fd_dft_600K_wave3.npz"
DFPT_LINE="$ROOT/results/graphene_physical_fd_dfpt/campaigns/FD600_LINE/graphene_FD600_LINE_dfpt.csv"
POOLED="$TD/td_graphene_v11_fd600_delta_weighted_short_pooled360.npz"
STATIC_FC2="$OUT/static_short_fc2.npz"
QSPACE="$OUT/qspace"
LOG="$OUT/run.log"
SSH=(env -u LD_LIBRARY_PATH ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
SCP=(env -u LD_LIBRARY_PATH scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)

cd "$ROOT"
mkdir -p "$OUT" "$TD" "$QSPACE" "$LOCAL_WEIGHTED/T600" "$LOCAL_WEIGHTED/holdout600"
exec > >(tee -a "$LOG") 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "=== weighted-delta 600 K TDEP START $(date -Is); phonon_env=$PHONON_ENV mlip_env=$MLIP_ENV ==="

if [[ -e "$OUT/DONE" ]]; then
    exit 0
fi
while [[ ! -e "$LOCAL_HOLDOUT/RELAYED_SHARD_A_TO_2060" ]]; do
    echo "waiting for local DFT holdout shard A to release V100-A: $(date -Is)"
    sleep 60
done
while ! "${SSH[@]}" "$REMOTE" "test -e '$REMOTE_WEIGHTED/holdout600/DONE'"; do
    echo "waiting for frozen model and independent force gate on 2060: $(date -Is)"
    sleep 60
done
if ! "${SSH[@]}" "$REMOTE" "test -e '$REMOTE_WEIGHTED/holdout600/INDEPENDENT_FORCE_GATE_PASSED'"; then
    touch "$OUT/SKIPPED_INDEPENDENT_FORCE_GATE" "$OUT/DONE"
    "${SSH[@]}" "$REMOTE" "touch '$REMOTE_WEIGHTED/TDEP600_SKIPPED_FORCE_GATE'"
    echo "independent force gate failed; 600 K TDEP was not started"
    exit 0
fi

fetch() {
    local remote_path="$1" local_path="$2"
    mkdir -p "$(dirname "$local_path")"
    "${SCP[@]}" "$REMOTE:$remote_path" "$local_path.partial"
    mv "$local_path.partial" "$local_path"
}

relay() {
    local local_path="$1" remote_path="$2"
    "${SSH[@]}" "$REMOTE" "mkdir -p '$(dirname "$remote_path")'"
    "${SCP[@]}" "$local_path" "$REMOTE:$remote_path.partial"
    "${SSH[@]}" "$REMOTE" "mv '$remote_path.partial' '$remote_path'"
}

fetch "$REMOTE_WEIGHTED/T600/selected_checkpoint.model" \
    "$LOCAL_WEIGHTED/T600/selected_checkpoint.model"
fetch "$REMOTE_WEIGHTED/T600/checkpoint_selection.json" \
    "$LOCAL_WEIGHTED/T600/checkpoint_selection.json"
fetch "$REMOTE_WEIGHTED/holdout600/gate_selection.json" \
    "$LOCAL_WEIGHTED/holdout600/gate_selection.json"
if [[ ! -s "$DFT" ]]; then
    fetch "$REMOTE_ROOT/results/td_phonon/graphene_physical_fd_dft_600K_wave3.npz" "$DFT"
fi
if [[ ! -s "$DFPT_LINE" ]]; then
    fetch "$REMOTE_ROOT/results/graphene_physical_fd_dfpt/campaigns/FD600_LINE/graphene_FD600_LINE_dfpt.csv" "$DFPT_LINE"
fi
MODEL="$LOCAL_WEIGHTED/T600/selected_checkpoint.model"
for path in "$MODEL" "$V11" "$OPERATOR" "$BG" "$DFT" "$DFPT_LINE"; do [[ -s "$path" ]]; done

run_seed() {
    local seed="$1" dt="$2" equil="$3" stride="$4"
    local tag="graphene_v11_fd600_delta_weighted_full_sampling_seed${seed}"
    [[ -s "$TD/td_${tag}.npz" && -s "$TD/td_${tag}.csv" ]] && return 0
    "$CONDA" run --no-capture-output -n "$MLIP_ENV" python \
        scripts/smearing_kink/td_phonon_friedel.py \
        --model "$V11" --delta-model "$MODEL" --bg "$BG" \
        --operator "$OPERATOR" --device cuda --tag "$tag" \
        --temperatures 600 --dt "$dt" --equil "$equil" \
        --nsnap 120 --stride "$stride" --npoints 201 --seed "$seed" \
        --checkpoint-root "results/td_phonon/${tag}_checkpoint" \
        --checkpoint-every 25 --max-temperature-factor 5 \
        --min-pair-distance 0.8 --max-force 100
}

# Equal physical equilibration/snapshot spacing; seed 1 independently tests 0.25 fs.
specifications=("0:0.5:3000:80" "1:0.25:6000:160" "2:0.5:3000:80")
for specification in "${specifications[@]}"; do
    IFS=: read -r seed dt equil stride <<< "$specification"
    if ! run_seed "$seed" "$dt" "$equil" "$stride"; then
        touch "$OUT/FAILED_STABILITY" "$OUT/DONE"
        relay "$LOG" "$REMOTE_WEIGHTED/T600_TDEP_run.log"
        "${SSH[@]}" "$REMOTE" "touch '$REMOTE_WEIGHTED/TDEP600_FAILED_STABILITY'"
        echo "600 K seed=$seed, dt=$dt fs failed trajectory health checks"
        exit 0
    fi
    stem="td_graphene_v11_fd600_delta_weighted_full_sampling_seed${seed}"
    relay "$TD/${stem}.npz" "$REMOTE_ROOT/results/td_phonon/${stem}.npz"
    relay "$TD/${stem}.csv" "$REMOTE_ROOT/results/td_phonon/${stem}.csv"
done

for seed in 0 1 2; do
    checkpoint="$TD/graphene_v11_fd600_delta_weighted_full_sampling_seed${seed}_checkpoint/T600"
    short_seed="$TD/td_graphene_v11_fd600_delta_weighted_short_seed${seed}.npz"
    "$CONDA" run --no-capture-output -n "$PHONON_ENV" python \
        scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
        --background "$BG" --checkpoint "$checkpoint" \
        --subtract-operator "$OPERATOR" --temperature 600 \
        --tag "graphene_v11_fd600_delta_weighted_short_seed${seed}" \
        --output "$short_seed"
    relay "$short_seed" "$REMOTE_ROOT/results/td_phonon/$(basename "$short_seed")"
done
"$CONDA" run --no-capture-output -n "$PHONON_ENV" python \
    scripts/smearing_kink/evaluate_graphene_fd_delta_weighted_tdep.py \
    --temperature 600 --dft-reference "$DFT" \
    --candidate "$TD/td_graphene_v11_fd600_delta_weighted_full_sampling_seed0.npz" \
    --candidate "$TD/td_graphene_v11_fd600_delta_weighted_full_sampling_seed1.npz" \
    --candidate "$TD/td_graphene_v11_fd600_delta_weighted_full_sampling_seed2.npz" \
    --force-selection "$LOCAL_WEIGHTED/holdout600/gate_selection.json" \
    --scope "600 K reconstructed full-force sampling diagnostic before replacing provisional LR in q-space" \
    --output "$OUT/full_sampling_tdep_diagnostic.json"
"$CONDA" run --no-capture-output -n "$PHONON_ENV" python \
    scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
    --background "$BG" \
    --checkpoint "$TD/graphene_v11_fd600_delta_weighted_full_sampling_seed0_checkpoint/T600" \
    --checkpoint "$TD/graphene_v11_fd600_delta_weighted_full_sampling_seed1_checkpoint/T600" \
    --checkpoint "$TD/graphene_v11_fd600_delta_weighted_full_sampling_seed2_checkpoint/T600" \
    --subtract-operator "$OPERATOR" \
    --temperature 600 --tag graphene_v11_fd600_delta_weighted_short_pooled360 \
    --output "$POOLED"
"$CONDA" run --no-capture-output -n "$MLIP_ENV" python \
    scripts/smearing_kink/build_graphene_fd_delta_static_fc2.py \
    --base-model "$V11" --delta-model "$MODEL" --background "$BG" \
    --device cuda --distance 0.01 --output "$STATIC_FC2" \
    --manifest "$OUT/static_short_fc2_manifest.json"
"$CONDA" run --no-capture-output -n "$PHONON_ENV" python \
    scripts/smearing_kink/evaluate_graphene_fd_finite_temperature_qspace.py \
    --temperature 600 --degauss 0.0038001738 \
    --dfpt-line "$DFPT_LINE" --background "$BG" \
    --static-short-fc2 "$STATIC_FC2" --thermal-short-tdep "$POOLED" \
    --dft-tdep "$DFT" \
    --thermal-seed "$TD/td_graphene_v11_fd600_delta_weighted_short_seed0.npz" \
    --thermal-seed "$TD/td_graphene_v11_fd600_delta_weighted_short_seed1.npz" \
    --thermal-seed "$TD/td_graphene_v11_fd600_delta_weighted_short_seed2.npz" \
    --force-selection "$LOCAL_WEIGHTED/holdout600/gate_selection.json" \
    --scope "600 K frozen model: independent 15-structure force holdout; full-force sampling with exact provisional-LR subtraction; three-seed short-TDEP plus q-space transfer" \
    --output-dir "$QSPACE"
cp -p "$QSPACE/acceptance.json" "$OUT/acceptance.json"
if "$CONDA" run -n "$PHONON_ENV" python -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["passes_all_force_seed_qspace_gates"] else 1)' \
    "$OUT/acceptance.json"; then
    touch "$OUT/PASSED_FORCE_SEED_QSPACE_GATE"
else
    touch "$OUT/BLOCKED_SEED_OR_QSPACE_GATE"
fi
relay "$OUT/acceptance.json" "$REMOTE_WEIGHTED/T600_TDEP/acceptance.json"
relay "$OUT/static_short_fc2_manifest.json" "$REMOTE_WEIGHTED/T600_TDEP/static_short_fc2_manifest.json"
relay "$OUT/full_sampling_tdep_diagnostic.json" "$REMOTE_WEIGHTED/T600_TDEP/full_sampling_tdep_diagnostic.json"
relay "$QSPACE/finite_temperature_qspace_predictions.csv" "$REMOTE_WEIGHTED/T600_TDEP/finite_temperature_qspace_predictions.csv"
relay "$QSPACE/finite_temperature_qspace_comparison.png" "$REMOTE_WEIGHTED/T600_TDEP/finite_temperature_qspace_comparison.png"
relay "$LOG" "$REMOTE_WEIGHTED/T600_TDEP/run.log"
touch "$OUT/DONE"
"${SSH[@]}" "$REMOTE" "touch '$REMOTE_WEIGHTED/TDEP600_REMOTE_DONE'"
echo "=== weighted-delta 600 K short-TDEP plus q-space LR COMPLETE $(date -Is) ==="
