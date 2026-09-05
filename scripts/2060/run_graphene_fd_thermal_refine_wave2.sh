#!/usr/bin/env bash
# Expanded physical-FD graphene fine-tune: two non-overlapping DFT label waves,
# balanced v11 harmonic replay, DFT-label TDEP references, held-out force
# validation, and three independent MLIP-TDEP seeds per temperature.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
SOURCE="$ROOT/data/graphene_fd_thermal_labels"
DATA="$ROOT/data/graphene_fd_thermal_finetune_wave2"
OUT="$ROOT/results/graphene_fd_thermal_finetune_wave2"
WAVE1_OUT="$ROOT/results/graphene_fd_thermal_finetune"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
DONE="$OUT/DONE"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$DATA" "$OUT" results/td_phonon
exec > >(tee -a "$LOG") 2>&1
echo "=== graphene physical-FD thermal refine wave2 queue start $(date -Is) ==="
while [[ ! -e "$SOURCE/.READY_WAVE2" || ! -e "$WAVE1_OUT/DONE" ]]; do
    echo "waiting for wave2 labels and wave1 fine-tune: $(date -Is)"
    sleep 60
done
for path in "$SOURCE/A.xyz" "$SOURCE/B.xyz" "$SOURCE/A2.xyz" "$SOURCE/B2.xyz" "$V11"; do
    [[ -s "$path" ]]
done
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/prepare_graphene_fd_thermal_finetune.py \
    --labels-a "$SOURCE/A.xyz" --labels-a "$SOURCE/A2.xyz" \
    --labels-b "$SOURCE/B.xyz" --labels-b "$SOURCE/B2.xyz" \
    --replay data/finetune_graphene8/train.xyz --replay-count 24 \
    --output "$DATA"

train_temperature() {
    local temperature="$1" name="gr_v11_fd${1}_wave2" lane_out="$OUT/T${1}"
    local model="$lane_out/${name}.model"
    mkdir -p "$lane_out/checkpoints" "$lane_out/logs" "$lane_out/results"
    if [[ -s "$model" ]]; then
        echo "T=$temperature wave2 model already complete: $model"
        return 0
    fi
    local restart=()
    if find "$lane_out/checkpoints" -type f -name '*.pt' -print -quit | grep -q .; then
        restart+=(--restart_latest)
    fi
    echo "--- T=$temperature expanded v11 physical-FD fine-tune $(date -Is) ---"
    "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
        --name "$name" --foundation_model "$V11" \
        --multiheads_finetuning False --foundation_model_elements True \
        --train_file "$DATA/T${temperature}/train.xyz" \
        --valid_file "$DATA/T${temperature}/val.xyz" \
        --test_file "$DATA/T${temperature}/val.xyz" \
        --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
        --energy_weight 0.0 --forces_weight 100.0 \
        --batch_size 1 --valid_batch_size 1 \
        --max_num_epochs 240 --patience 50 --eval_interval 2 \
        --lr 0.0003 --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
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
    --model "v11=$V11" \
    --model "fd300=$OUT/T300/gr_v11_fd300_wave2.model" \
    --model "fd600=$OUT/T600/gr_v11_fd600_wave2.model" \
    --device cuda --output "$OUT/force_validation.json"

# A failed held-out force gate cannot be repaired by the later TDEP fit.  Write
# an early machine-readable decision so the idle V100 GPUs can start wave 3
# while this service still completes all wave-2 TDEP diagnostics.
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/write_graphene_physical_fd_force_gate.py \
    --force-validation "$OUT/force_validation.json" \
    --output "$OUT/acceptance.json"

fit_dft_reference() {
    local temperature="$1" degauss="$2" lane="$3" wave="$4"
    local extra=()
    if [[ "$wave" == wave2 ]]; then
        extra+=(--labels "$SOURCE/${lane}2.xyz")
    fi
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/fit_graphene_physical_fd_tdep_labels.py \
        --labels "$SOURCE/${lane}.xyz" "${extra[@]}" \
        --temperature "$temperature" --degauss "$degauss" \
        --output "results/td_phonon/graphene_physical_fd_dft_${temperature}K_${wave}.npz"
}
fit_dft_reference 300 0.0019000869 A wave1
fit_dft_reference 600 0.0038001738 B wave1
fit_dft_reference 300 0.0019000869 A wave2
fit_dft_reference 600 0.0038001738 B wave2

run_tdep_seed() {
    local temperature="$1" seed="$2"
    local model="$OUT/T${temperature}/gr_v11_fd${temperature}_wave2.model"
    local tag="graphene_v11_fd${temperature}_wave2_short_range_seed${seed}"
    if [[ -s "results/td_phonon/td_${tag}.npz" && -s "results/td_phonon/td_${tag}.csv" ]]; then
        echo "T=$temperature seed=$seed TDEP already complete"
        return 0
    fi
    local attempts=("1.0:1500:40:dt1")
    if [[ "$temperature" == 600 ]]; then
        # Keep 1.5 ps equilibration and 40 fs snapshot spacing while reducing
        # the integration step.  The second setting is an automatic fallback.
        attempts=("0.5:3000:80:dt0p5" "0.25:6000:160:dt0p25")
    fi
    local specification dt equil stride suffix
    for specification in "${attempts[@]}"; do
        IFS=: read -r dt equil stride suffix <<< "$specification"
        echo "T=$temperature seed=$seed TDEP attempt dt=$dt fs"
        if "$CONDA" run --no-capture-output -n phonon python scripts/td_phonon.py \
            --structure data/td_phonon/graphene.xyz --model "$model" --device cuda \
            --tag "$tag" --outdir results/td_phonon --temperatures "$temperature" \
            --supercell 6,6,1 --no-relax --a 2.4600000087 \
            --dt "$dt" --equil "$equil" --nsnap 120 --stride "$stride" \
            --npoints 201 --seed "$seed" \
            --checkpoint-root "results/td_phonon/${tag}_${suffix}_checkpoint" \
            --checkpoint-every 25 --max-temperature-factor 5 \
            --min-pair-distance 0.8 --max-force 100; then
            return 0
        fi
        echo "T=$temperature seed=$seed failed stability check at dt=$dt fs"
    done
    return 1
}
for temperature in 300 600; do
    for seed in 0 1 2; do
        run_tdep_seed "$temperature" "$seed"
    done
done

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_physical_fd_tdep.py \
    --root results/td_phonon --force-validation "$OUT/force_validation.json" \
    --output "$OUT/acceptance.json"

touch "$DONE"
echo "=== graphene physical-FD thermal refine wave2 COMPLETE $(date -Is) ==="
