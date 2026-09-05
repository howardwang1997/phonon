#!/usr/bin/env bash
# Recompute both endpoint TDEPs with the frozen joint model and fit the T law.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
RESULTS="$ROOT/results/graphene_fd_transferability"
P2="$RESULTS/P2_joint_short"
SHORT="$RESULTS/short_model"
ENDPOINTS="$RESULTS/endpoints"
TD="$ROOT/results/td_phonon"
DATA="$ROOT/data/graphene_fd_delta_pilot"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
MODEL="$P2/selected_checkpoint.model"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
FORCE_GATE="$SHORT/joint_endpoint_force_gate.json"
STATIC="$SHORT/static_short_fc2.npz"
STATIC_MANIFEST="$SHORT/static_short_fc2_manifest.json"
LAW="$RESULTS/temperature_law.json"
LOG="$RESULTS/P3_run.log"

cd "$ROOT"
mkdir -p "$RESULTS" "$SHORT" "$ENDPOINTS" "$TD"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$RESULTS/P3_DONE" ]]; then
    echo "P3 endpoint TDEP and temperature-law stage already complete"
    exit 0
fi
while [[ ! -e "$P2/DONE" ]]; do
    echo "waiting for P2 joint endpoint selection: $(date -Is)"
    sleep 60
done
if [[ ! -e "$P2/PASSED_ENDPOINT_DEVELOPMENT_GATE" ]]; then
    touch "$RESULTS/BLOCKED_P2_ENDPOINT_GATE" "$RESULTS/P3_DONE"
    echo "P2 did not pass both endpoint gates; no TDEP or 450 K work started"
    exit 0
fi
for path in "$V11" "$MODEL" "$BG" "$P2/checkpoint_selection.json" \
    "$DATA/T300/long_range_operator.npz" \
    "$DATA/T600/long_range_operator.npz"; do
    [[ -s "$path" ]]
done

if [[ ! -s "$FORCE_GATE" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/materialize_graphene_fd_joint_force_gate.py \
        --selection "$P2/checkpoint_selection.json" --model "$MODEL" \
        --output "$FORCE_GATE"
fi
if [[ ! -s "$STATIC" || ! -s "$STATIC_MANIFEST" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/build_graphene_fd_delta_static_fc2.py \
        --base-model "$V11" --delta-model "$MODEL" --background "$BG" \
        --device cuda --output "$STATIC" --manifest "$STATIC_MANIFEST"
fi

run_seed() {
    local temperature="$1"
    local seed="$2"
    local operator="$3"
    local endpoint="$4"
    local tag="graphene_fd_transferability_joint_T${temperature}_full_seed${seed}"
    local short_tdep="$endpoint/short_seed${seed}.npz"
    local checkpoint_record="$endpoint/seed${seed}_checkpoint_path.txt"
    if [[ -s "$short_tdep" && -s "$checkpoint_record" ]]; then
        local recorded_checkpoint
        recorded_checkpoint="$(<"$checkpoint_record")"
        [[ -s "$recorded_checkpoint/snapshots.npz" ]]
        return 0
    fi

    local specifications=()
    if [[ "$temperature" == 300 ]]; then
        specifications=("1.0:1500:40:dt1" "0.5:3000:80:dt0p5")
    else
        specifications=("0.5:3000:80:dt0p5" "0.25:6000:160:dt0p25")
    fi
    local specification dt equil stride suffix checkpoint_root checkpoint_path
    for specification in "${specifications[@]}"; do
        IFS=: read -r dt equil stride suffix <<< "$specification"
        checkpoint_root="$endpoint/checkpoints/seed${seed}_${suffix}"
        checkpoint_path="$checkpoint_root/T${temperature}"
        echo "T=$temperature seed=$seed endpoint sampling attempt dt=$dt fs"
        if "$CONDA" run --no-capture-output -n phonon python \
            scripts/smearing_kink/td_phonon_friedel.py \
            --model "$V11" --delta-model "$MODEL" --bg "$BG" \
            --operator "$operator" --device cuda --tag "$tag" \
            --temperatures "$temperature" --dt "$dt" --equil "$equil" \
            --nsnap 120 --stride "$stride" --npoints 201 --seed "$seed" \
            --checkpoint-root "$checkpoint_root" --checkpoint-every 25 \
            --max-temperature-factor 5 --min-pair-distance 0.8 \
            --max-force 100; then
            printf '%s\n' "$checkpoint_path" > "$checkpoint_record"
            "$CONDA" run --no-capture-output -n phonon python \
                scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
                --background "$BG" --checkpoint "$checkpoint_path" \
                --subtract-operator "$operator" --temperature "$temperature" \
                --tag "graphene_fd_transferability_joint_T${temperature}_short_seed${seed}" \
                --output "$short_tdep"
            return 0
        fi
    done
    return 1
}

run_endpoint() {
    local temperature="$1"
    local degauss operator dfpt dft
    case "$temperature" in
        300)
            degauss=0.0019000869
            operator="$DATA/T300/long_range_operator.npz"
            dfpt="$ROOT/results/graphene_physical_fd_dfpt/campaigns/FD300_LINE/graphene_FD300_LINE_dfpt.csv"
            dft="$TD/graphene_physical_fd_dft_300K_wave3.npz"
            ;;
        600)
            degauss=0.0038001738
            operator="$DATA/T600/long_range_operator.npz"
            dfpt="$ROOT/results/graphene_physical_fd_dfpt/campaigns/FD600_LINE/graphene_FD600_LINE_dfpt.csv"
            dft="$TD/graphene_physical_fd_dft_600K_wave3.npz"
            ;;
        *) return 2 ;;
    esac
    local endpoint="$ENDPOINTS/T${temperature}"
    mkdir -p "$endpoint/checkpoints"
    for path in "$operator" "$dfpt" "$dft"; do [[ -s "$path" ]]; done

    local seed
    for seed in 0 1 2; do
        if ! run_seed "$temperature" "$seed" "$operator" "$endpoint"; then
            touch "$endpoint/FAILED_STABILITY"
            echo "T=$temperature seed=$seed failed both predefined timesteps"
            return 0
        fi
    done

    local checkpoints=()
    local seeds=()
    for seed in 0 1 2; do
        checkpoints+=("$(<"$endpoint/seed${seed}_checkpoint_path.txt")")
        seeds+=("$endpoint/short_seed${seed}.npz")
    done
    local pooled="$endpoint/short_pooled360.npz"
    if [[ ! -s "$pooled" ]]; then
        local pooled_args=()
        local checkpoint
        for checkpoint in "${checkpoints[@]}"; do
            pooled_args+=(--checkpoint "$checkpoint")
        done
        "$CONDA" run --no-capture-output -n phonon python \
            scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
            --background "$BG" "${pooled_args[@]}" \
            --subtract-operator "$operator" --temperature "$temperature" \
            --tag "graphene_fd_transferability_joint_T${temperature}_short_pooled360" \
            --output "$pooled"
    fi

    local tdep_manifest="$endpoint/tdep_manifest.json"
    if [[ ! -s "$tdep_manifest" ]]; then
        local summary_args=()
        local checkpoint seed_tdep
        for checkpoint in "${checkpoints[@]}"; do
            summary_args+=(--checkpoint "$checkpoint")
        done
        for seed_tdep in "${seeds[@]}"; do
            summary_args+=(--seed-tdep "$seed_tdep")
        done
        "$CONDA" run --no-capture-output -n phonon python \
            scripts/smearing_kink/summarize_graphene_fd_joint_tdep.py \
            --temperature "$temperature" --degauss "$degauss" \
            --base-model "$V11" --delta-model "$MODEL" --background "$BG" \
            --operator "$operator" --static-manifest "$STATIC_MANIFEST" \
            --force-gate "$FORCE_GATE" "${summary_args[@]}" \
            --pooled-tdep "$pooled" --mean-temperature-relative-threshold 0.05 \
            --output "$tdep_manifest"
    fi
    if ! "$CONDA" run -n phonon python -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["passes_tdep_sampling_gate"] else 1)' \
        "$tdep_manifest"; then
        touch "$endpoint/BLOCKED_TDEP_SAMPLING_GATE"
        echo "T=$temperature TDEP sampling gate failed; q-space calibration skipped"
        return 0
    fi

    local static_transfer="$endpoint/qspace_transfer"
    if [[ ! -s "$static_transfer/acceptance.json" ]]; then
        mkdir -p "$static_transfer"
        "$CONDA" run --no-capture-output -n phonon python \
            scripts/smearing_kink/evaluate_graphene_fd_finite_temperature_qspace.py \
            --temperature "$temperature" --degauss "$degauss" \
            --dfpt-line "$dfpt" --background "$BG" \
            --static-short-fc2 "$STATIC" --thermal-short-tdep "$pooled" \
            --dft-tdep "$dft" \
            --thermal-seed "${seeds[0]}" --thermal-seed "${seeds[1]}" \
            --thermal-seed "${seeds[2]}" --force-selection "$FORCE_GATE" \
            --mean-temperature-relative-threshold 0.05 \
            --scope "$temperature K endpoint development control with one frozen joint short model" \
            --output-dir "$static_transfer"
    fi

    local calibrated="$endpoint/qspace_calibrated"
    if [[ ! -s "$calibrated/acceptance.json" ]]; then
        mkdir -p "$calibrated"
        "$CONDA" run --no-capture-output -n phonon python \
            scripts/smearing_kink/evaluate_graphene_fd_finite_temperature_calibrated_qspace.py \
            --temperature "$temperature" --degauss "$degauss" \
            --dfpt-line "$dfpt" --background "$BG" \
            --thermal-short-tdep "$pooled" --dft-tdep "$dft" \
            --thermal-seed "${seeds[0]}" --thermal-seed "${seeds[1]}" \
            --thermal-seed "${seeds[2]}" --force-selection "$FORCE_GATE" \
            --static-transfer-acceptance "$static_transfer/acceptance.json" \
            --mean-temperature-relative-threshold 0.05 \
            --scope "$temperature K endpoint development calibration for the frozen joint short model" \
            --output-dir "$calibrated"
    fi
    if "$CONDA" run -n phonon python -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["passes_all_force_seed_calibrated_qspace_gates"] else 1)' \
        "$calibrated/acceptance.json"; then
        touch "$endpoint/PASSED_ENDPOINT_GATE"
        echo "T=$temperature joint-model endpoint gate passed"
    else
        touch "$endpoint/BLOCKED_ENDPOINT_GATE"
        echo "T=$temperature calibrated endpoint gate failed"
    fi
}

echo "=== graphene transferability P3 START $(date -Is) ==="
run_endpoint 300
if [[ ! -e "$ENDPOINTS/T300/PASSED_ENDPOINT_GATE" ]]; then
    touch "$RESULTS/BLOCKED_P3_ENDPOINT_GATE" "$RESULTS/P3_DONE"
    echo "P3 stopped at the 300 K endpoint; no 450 K labels were started"
    exit 0
fi
run_endpoint 600
if [[ ! -e "$ENDPOINTS/T600/PASSED_ENDPOINT_GATE" ]]; then
    touch "$RESULTS/BLOCKED_P3_ENDPOINT_GATE" "$RESULTS/P3_DONE"
    echo "P3 stopped at the 600 K endpoint; no 450 K labels were started"
    exit 0
fi

if [[ ! -s "$LAW" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/fit_graphene_fd_temperature_law.py \
        --endpoint "300=$ENDPOINTS/T300/qspace_calibrated/acceptance.json" \
        --endpoint "600=$ENDPOINTS/T600/qspace_calibrated/acceptance.json" \
        --tdep-manifest "300=$ENDPOINTS/T300/tdep_manifest.json" \
        --tdep-manifest "600=$ENDPOINTS/T600/tdep_manifest.json" \
        --force-gate "$FORCE_GATE" --base-model "$V11" --delta-model "$MODEL" \
        --prediction-temperature 450 --output "$LAW"
fi
touch "$RESULTS/READY_FOR_FREEZE" "$RESULTS/P3_DONE"
echo "=== graphene transferability P3 COMPLETE $(date -Is) ==="
