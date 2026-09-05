#!/usr/bin/env bash
# Conditional final data-expansion pass.  No fourth wave is launched
# automatically; a failure here is treated as a model/representation issue.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
SOURCE="$ROOT/data/graphene_fd_thermal_labels"
DATA="$ROOT/data/graphene_fd_thermal_finetune_wave3"
OUT="$ROOT/results/graphene_fd_thermal_finetune_wave3"
WAVE2_OUT="$ROOT/results/graphene_fd_thermal_finetune_wave2"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$DATA" "$OUT" results/td_phonon
exec > >(tee -a "$LOG") 2>&1
echo "=== graphene physical-FD thermal refine wave3 queue start $(date -Is) ==="
if [[ -e "$OUT/FAILED_GATE" ]]; then
    echo "final short-range gate already failed; no unstable trajectory will be reused"
    exit 0
fi
while [[ ! -s "$WAVE2_OUT/acceptance.json" ]]; do
    echo "waiting for wave2 acceptance: $(date -Is)"
    sleep 300
done
wave2_pass="$($CONDA run -n phonon python -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["passes_short_range_gate"]))' \
    "$WAVE2_OUT/acceptance.json" | tail -1)"
if [[ "$wave2_pass" == 1 ]]; then
    touch "$OUT/SKIPPED_GATE_PASSED"
    echo "wave2 passed; 2060 wave3 refine skipped"
    exit 0
fi
while [[ ! -e "$SOURCE/.READY_WAVE3" ]]; do
    echo "wave2 failed; waiting for conditional wave3 labels: $(date -Is)"
    sleep 300
done
while [[ ! -e "$WAVE2_OUT/DONE" ]]; do
    echo "wave3 labels ready; waiting for wave2 diagnostics to release GPU: $(date -Is)"
    sleep 60
done
for suffix in A B A2 B2 A3 B3; do [[ -s "$SOURCE/$suffix.xyz" ]]; done
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/prepare_graphene_fd_thermal_finetune.py \
    --labels-a "$SOURCE/A.xyz" --labels-a "$SOURCE/A2.xyz" --labels-a "$SOURCE/A3.xyz" \
    --labels-b "$SOURCE/B.xyz" --labels-b "$SOURCE/B2.xyz" --labels-b "$SOURCE/B3.xyz" \
    --replay data/finetune_graphene8/train.xyz --replay-count 36 --output "$DATA"

train_temperature() {
    local temperature="$1" name="gr_v11_fd${1}_wave3" lane_out="$OUT/T${1}"
    local model="$lane_out/${name}.model"
    mkdir -p "$lane_out/checkpoints" "$lane_out/logs" "$lane_out/results"
    [[ -s "$model" ]] && return 0
    local restart=()
    if find "$lane_out/checkpoints" -type f -name '*.pt' -print -quit | grep -q .; then
        restart+=(--restart_latest)
    fi
    "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
        --name "$name" --foundation_model "$V11" \
        --multiheads_finetuning False --foundation_model_elements True \
        --train_file "$DATA/T${temperature}/train.xyz" \
        --valid_file "$DATA/T${temperature}/val.xyz" \
        --test_file "$DATA/T${temperature}/val.xyz" \
        --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
        --energy_weight 0.0 --forces_weight 100.0 --batch_size 1 --valid_batch_size 1 \
        --max_num_epochs 280 --patience 60 --eval_interval 2 \
        --lr 0.0002 --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
        --default_dtype float32 --device cuda --seed 29 --save_cpu \
        --model_dir "$lane_out" --checkpoints_dir "$lane_out/checkpoints" \
        --log_dir "$lane_out/logs" --results_dir "$lane_out/results" \
        "${restart[@]}"
    [[ -s "$model" ]]
}
train_temperature 300
train_temperature 600

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_thermal_force_models.py \
    --validation-a "$DATA/T300/val.xyz" --validation-b "$DATA/T600/val.xyz" \
    --model "v11=$V11" --model "fd300=$OUT/T300/gr_v11_fd300_wave3.model" \
    --model "fd600=$OUT/T600/gr_v11_fd600_wave3.model" \
    --device cuda --output "$OUT/force_validation.json"

fit_reference() {
    local temperature="$1" degauss="$2" lane="$3"
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/fit_graphene_physical_fd_tdep_labels.py \
        --labels "$SOURCE/${lane}.xyz" --labels "$SOURCE/${lane}2.xyz" \
        --labels "$SOURCE/${lane}3.xyz" --temperature "$temperature" \
        --degauss "$degauss" \
        --output "results/td_phonon/graphene_physical_fd_dft_${temperature}K_wave3.npz"
}
fit_reference 300 0.0019000869 A
fit_reference 600 0.0038001738 B

run_seed() {
    local temperature="$1" seed="$2"
    local tag="graphene_v11_fd${temperature}_wave3_short_range_seed${seed}"
    [[ -s "results/td_phonon/td_${tag}.npz" && -s "results/td_phonon/td_${tag}.csv" ]] && return 0
    local attempts=("1.0:1500:40:dt1")
    if [[ "$temperature" == 600 ]]; then
        attempts=("0.5:3000:80:dt0p5" "0.25:6000:160:dt0p25")
    fi
    local specification dt equil stride suffix
    for specification in "${attempts[@]}"; do
        IFS=: read -r dt equil stride suffix <<< "$specification"
        if "$CONDA" run --no-capture-output -n phonon python scripts/td_phonon.py \
            --structure data/td_phonon/graphene.xyz \
            --model "$OUT/T${temperature}/gr_v11_fd${temperature}_wave3.model" \
            --device cuda --tag "$tag" --outdir results/td_phonon \
            --temperatures "$temperature" --supercell 6,6,1 --no-relax \
            --a 2.4600000087 --dt "$dt" --equil "$equil" --nsnap 120 \
            --stride "$stride" --npoints 201 --seed "$seed" \
            --checkpoint-root "results/td_phonon/${tag}_${suffix}_checkpoint" \
            --checkpoint-every 25 --max-temperature-factor 5 \
            --min-pair-distance 0.8 --max-force 100; then
            return 0
        fi
    done
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/write_graphene_physical_fd_force_gate.py \
        --force-validation "$OUT/force_validation.json" \
        --output "$OUT/acceptance.json" --final-wave \
        --trajectory-error "T=${temperature} K seed=${seed}: all configured timestep attempts failed trajectory stability checks"
    touch "$OUT/FAILED_GATE"
    return 1
}
for temperature in 300 600; do
    for seed in 0 1 2; do run_seed "$temperature" "$seed"; done
done

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_physical_fd_tdep.py \
    --root results/td_phonon --force-validation "$OUT/force_validation.json" \
    --wave wave3 --previous-wave wave2 --model-tag wave3 --final-wave \
    --output "$OUT/acceptance.json"
touch "$OUT/DONE"
echo "=== graphene physical-FD thermal refine wave3 COMPLETE $(date -Is) ==="
