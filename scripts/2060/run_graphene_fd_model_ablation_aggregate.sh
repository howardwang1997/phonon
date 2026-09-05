#!/usr/bin/env bash
# Evaluate replay/training variants, select fixed-gate winners, and only then
# run formal three-seed TDEP validation.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DATA="$ROOT/data/graphene_fd_model_ablation"
OUT="$ROOT/results/graphene_fd_model_ablation"
TD="$ROOT/results/td_phonon"
LOG="$OUT/aggregate.log"

cd "$ROOT"
mkdir -p "$OUT" "$TD"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while [[ ! -e "$OUT/T300/DONE" || ! -e "$OUT/T600/REMOTE_DONE" ]]; do
    echo "waiting for 300/600 K ablation training lanes: $(date -Is)"
    sleep 60
done
while [[ ! -e "$OUT/PRE300_DONE" \
      && ! -e "$OUT/PRE300_BLOCKED_FORCE_GATE" \
      && ! -e "$OUT/PRE300_STABILITY_FAILED" ]]; do
    echo "both model lanes complete; waiting for 300 K prevalidation: $(date -Is)"
    sleep 60
done

model_args=(
    --model "v11=$ROOT/results/gr_backbone_v11/ft_graphene.model"
    --model "balanced_300=$ROOT/results/graphene_fd_thermal_finetune_wave3/T300/gr_v11_fd300_wave3.model"
    --model "balanced_600=$ROOT/results/graphene_fd_thermal_finetune_wave3/T600/gr_v11_fd600_wave3.model"
)
for temperature in 300 600; do
    for variant in replay12 thermal_only continued; do
        model_args+=(--model "${variant}_${temperature}=$OUT/T${temperature}/${variant}/gr_fd${temperature}_${variant}.model")
    done
done

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_model_variants.py \
    --dataset "thermal300=$DATA/replay0/T300/test.xyz" \
    --dataset "thermal600=$DATA/replay0/T600/test.xyz" \
    --dataset "harmonic=data/finetune_graphene8/val.xyz" \
    "${model_args[@]}" --device cuda --output "$OUT/variant_force_metrics.json"

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/select_graphene_model_variants.py \
    --metrics "$OUT/variant_force_metrics.json" \
    --candidate-300 replay12_300 --candidate-300 thermal_only_300 \
    --candidate-300 continued_300 \
    --candidate-600 replay12_600 --candidate-600 thermal_only_600 \
    --candidate-600 continued_600 \
    --output "$OUT/selection.json"

passes="$($CONDA run -n phonon python -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["passes_both_temperature_gates"]))' \
    "$OUT/selection.json" | tail -1)"
if [[ "$passes" != 1 ]]; then
    touch "$OUT/BLOCKED_FORCE_GATE" "$OUT/DONE"
    echo "no candidate passed both thermal-force and harmonic-replay gates"
    exit 0
fi

model300="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["by_temperature"]["300"]["selected"]["path"])' \
    "$OUT/selection.json" | tail -1)"
model600="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["by_temperature"]["600"]["selected"]["path"])' \
    "$OUT/selection.json" | tail -1)"

if [[ -s "$OUT/selection_300.json" ]]; then
    pre300="$($CONDA run -n phonon python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["by_temperature"]["300"]["selected"]["path"])' \
        "$OUT/selection_300.json" | tail -1)"
    if [[ "$pre300" != "$model300" ]]; then
        touch "$OUT/BLOCKED_SELECTION_MISMATCH" "$OUT/DONE"
        echo "300 K early and final selections disagree; existing TDEP will not be reused"
        exit 0
    fi
fi

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_thermal_force_models.py \
    --validation-a "$DATA/replay0/T300/test.xyz" \
    --validation-b "$DATA/replay0/T600/test.xyz" \
    --model "v11=$ROOT/results/gr_backbone_v11/ft_graphene.model" \
    --model "fd300=$model300" --model "fd600=$model600" \
    --device cuda --output "$OUT/force_validation.json"

run_seed() {
    local temperature="$1" seed="$2" model="$3"
    local tag="graphene_v11_fd${temperature}_ablation_short_range_seed${seed}"
    [[ -s "$TD/td_${tag}.npz" && -s "$TD/td_${tag}.csv" ]] && return 0
    local attempts=("1.0:1500:40:dt1")
    if [[ "$temperature" == 600 ]]; then
        attempts=("0.5:3000:80:dt0p5" "0.25:6000:160:dt0p25")
    fi
    local specification dt equil stride suffix
    for specification in "${attempts[@]}"; do
        IFS=: read -r dt equil stride suffix <<< "$specification"
        if "$CONDA" run --no-capture-output -n phonon python scripts/td_phonon.py \
            --structure data/td_phonon/graphene.xyz --model "$model" --device cuda \
            --tag "$tag" --outdir results/td_phonon --temperatures "$temperature" \
            --supercell 6,6,1 --no-relax --a 2.4600000087 --dt "$dt" \
            --equil "$equil" --nsnap 120 --stride "$stride" --npoints 201 \
            --seed "$seed" --checkpoint-root "results/td_phonon/${tag}_${suffix}_checkpoint" \
            --checkpoint-every 25 --max-temperature-factor 5 \
            --min-pair-distance 0.8 --max-force 100; then
            return 0
        fi
    done
    return 1
}

for seed in 0 1 2; do
    if ! run_seed 300 "$seed" "$model300"; then
        "$CONDA" run --no-capture-output -n phonon python \
            scripts/smearing_kink/write_graphene_physical_fd_force_gate.py \
            --force-validation "$OUT/force_validation.json" \
            --output "$OUT/acceptance.json" --final-wave \
            --trajectory-error "T=300 K seed=${seed}: all configured timestep attempts failed"
        touch "$OUT/BLOCKED_STABILITY" "$OUT/DONE"
        exit 0
    fi
done

while [[ ! -e "$OUT/TDEP600_REMOTE_DONE" && ! -e "$OUT/TDEP600_FAILED" ]]; do
    echo "300 K TDEP complete; waiting for parallel V100-B 600 K lane: $(date -Is)"
    sleep 60
done
if [[ -e "$OUT/TDEP600_FAILED" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/write_graphene_physical_fd_force_gate.py \
        --force-validation "$OUT/force_validation.json" \
        --output "$OUT/acceptance.json" --final-wave \
        --trajectory-error "T=600 K: V100-B trajectory failed all configured timestep attempts"
    touch "$OUT/BLOCKED_STABILITY" "$OUT/DONE"
    exit 0
fi

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_physical_fd_tdep.py \
    --root results/td_phonon --force-validation "$OUT/force_validation.json" \
    --wave wave3 --previous-wave wave2 --model-tag ablation --final-wave \
    --high-symmetry-threshold 50 \
    --output "$OUT/acceptance.json"

passed_final="$($CONDA run -n phonon python -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["passes_short_range_gate"]))' \
    "$OUT/acceptance.json" | tail -1)"
if [[ "$passed_final" == 1 ]]; then
    touch "$OUT/PASSED_SHORT_RANGE"
else
    touch "$OUT/BLOCKED_TDEP_GATE"
fi
touch "$OUT/DONE"
echo "=== graphene model ablation aggregate COMPLETE $(date -Is) ==="
