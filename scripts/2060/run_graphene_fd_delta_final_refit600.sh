#!/usr/bin/env bash
# Final refit after development-only replay weight and epoch selection.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DATA="$ROOT/data/graphene_fd_delta_pilot/T600"
WEIGHTED="$ROOT/results/graphene_fd_delta_weighted/T600"
OUT="$WEIGHTED/final_refit"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
HARMONIC="$ROOT/data/finetune_graphene8/val.xyz"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT/checkpoints" "$OUT/logs" "$OUT/results"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

while [[ ! -e "$WEIGHTED/FINAL_RETRAIN_REQUIRED" ]]; do
    if [[ -e "$WEIGHTED/CHECKPOINT_SELECTION_DONE" ]]; then
        echo "a development checkpoint passed; final refit not required"
        exit 0
    fi
    echo "waiting for development checkpoint decision: $(date -Is)"
    sleep 60
done
if [[ -e "$WEIGHTED/FINAL_RETRAIN_DONE" ]]; then
    exit 0
fi

tag="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["tag"])' \
    "$WEIGHTED/checkpoint_selection.json" | tail -1)"
epoch="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["selected_epoch"])' \
    "$WEIGHTED/checkpoint_selection.json" | tail -1)"
case "$tag" in
    replayw4) weight=4 ;;
    replayw8) weight=8 ;;
    replayw12) weight=12 ;;
    replayw16) weight=16 ;;
    *) echo "unsupported selected tag: $tag"; exit 2 ;;
esac
max_epochs=$((epoch + 1))
name="gr_fd600_delta32_final_${tag}_epoch${epoch}"
training_model="$OUT/${name}.model"
model="$OUT/${name}_frozen_epoch.model"

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/prepare_graphene_fd_delta_final_refit.py \
    --train "$DATA/train.xyz" --development-validation "$DATA/val.xyz" \
    --development-thermal "$DATA/test.xyz" \
    --thermal-repeat 2 --output "$OUT/train.xyz" \
    --manifest "$OUT/data_manifest.json"
if [[ ! -s "$training_model" ]]; then
    "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
        --name "$name" --model MACE --num_interactions 2 \
        --hidden_irreps '32x0e+32x1o' --r_max 5.0 --num_radial_basis 8 \
        --num_cutoff_basis 5 --correlation 2 \
        --train_file "$OUT/train.xyz" --valid_file "$DATA/val.xyz" \
        --test_file "$DATA/test.xyz" \
        --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
        --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
        --config_type_weights "{'physical_fd_thermal_600K':1.0,'v11_fc_distillation_replay':${weight}.0,'physical_fd_thermal_600K_validation':1.0,'Default':${weight}.0}" \
        --batch_size 1 --valid_batch_size 1 --max_num_epochs "$max_epochs" \
        --patience "$((max_epochs + 5))" --eval_interval 5 --lr 0.001 \
        --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
        --default_dtype float32 --device cuda --seed 83 --save_cpu \
        --keep_checkpoints --save_all_checkpoints \
        --model_dir "$OUT" --checkpoints_dir "$OUT/checkpoints" \
        --log_dir "$OUT/logs" --results_dir "$OUT/results"
fi
[[ -s "$training_model" ]]
if [[ ! -s "$model" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/evaluate_graphene_fd_delta_checkpoints.py \
        --base-model "$V11" --template-model "$training_model" \
        --checkpoint-dir "$OUT/checkpoints" --only-epoch "$epoch" \
        --thermal "$DATA/test.xyz" --harmonic "$HARMONIC" --device cuda \
        --scope "exact frozen epoch after final refit; thermal metric is in-sample" \
        --output "$OUT/exact_checkpoint_metrics.json" \
        --selected-model "$model"
fi
[[ -s "$model" ]]
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_fd_delta_model.py \
    --dataset "thermal600_refit_in_sample=$DATA/test.xyz" \
    --dataset "harmonic=$HARMONIC" --base-model "$V11" \
    --delta-model "$model" --device cuda --output "$OUT/gate_metrics.json"
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/select_graphene_fd_delta_model.py \
    --metrics "$OUT/gate_metrics.json" \
    --thermal-label thermal600_refit_in_sample \
    --output "$OUT/gate_selection.json"
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/finalize_graphene_fd_delta_refit.py \
    --checkpoint-selection "$WEIGHTED/checkpoint_selection.json" \
    --data-manifest "$OUT/data_manifest.json" \
    --gate-selection "$OUT/gate_selection.json" --model "$model" \
    --replay-weight "$weight" --epoch "$epoch" \
    --output "$OUT/final_refit_selection.json"
cp -p "$model" "$WEIGHTED/selected_checkpoint.model"
printf 'final_refit_%s_epoch%s\n' "$tag" "$epoch" > "$WEIGHTED/selected_checkpoint_tag.txt"
touch "$WEIGHTED/FINAL_RETRAIN_DONE" "$WEIGHTED/CHECKPOINT_SELECTION_DONE"
echo "=== final 600 K delta refit frozen before independent holdout $(date -Is) ==="
