#!/usr/bin/env bash
# Refine the log-spaced replay sweep at weight 12, still using development data only.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DATA="$ROOT/data/graphene_fd_delta_pilot/T600"
OUT="$ROOT/results/graphene_fd_delta_weighted/T600"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
HARMONIC="$ROOT/data/finetune_graphene8/val.xyz"
LANE="$OUT/replayw12"
NAME=gr_fd600_delta32_replayw12
MODEL="$LANE/${NAME}.model"
LOG="$OUT/replayw12_run.log"

cd "$ROOT"
mkdir -p "$LANE/checkpoints" "$LANE/logs" "$LANE/results"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

while [[ ! -e "$OUT/DONE" ]]; do
    echo "waiting for log-spaced replay sweep before weight-12 refinement: $(date -Is)"
    sleep 60
done
if [[ -e "$OUT/W12_DONE" ]]; then
    exit 0
fi
for path in "$V11" "$DATA/train.xyz" "$DATA/val.xyz" "$DATA/test.xyz" "$HARMONIC"; do
    [[ -s "$path" ]]
done

if [[ ! -s "$MODEL" ]]; then
    "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
        --name "$NAME" --model MACE --num_interactions 2 \
        --hidden_irreps '32x0e+32x1o' --r_max 5.0 --num_radial_basis 8 \
        --num_cutoff_basis 5 --correlation 2 \
        --train_file "$DATA/train.xyz" --valid_file "$DATA/val.xyz" \
        --test_file "$DATA/test.xyz" \
        --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
        --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
        --config_type_weights "{'physical_fd_thermal_600K':1.0,'v11_fc_distillation_replay':12.0,'physical_fd_thermal_600K_validation':1.0,'Default':12.0}" \
        --batch_size 1 --valid_batch_size 1 --max_num_epochs 180 \
        --patience 35 --eval_interval 5 --lr 0.001 \
        --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
        --default_dtype float32 --device cuda --seed 83 --save_cpu \
        --keep_checkpoints --save_all_checkpoints \
        --model_dir "$LANE" --checkpoints_dir "$LANE/checkpoints" \
        --log_dir "$LANE/logs" --results_dir "$LANE/results"
fi
[[ -s "$MODEL" ]]
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_fd_delta_model.py \
    --dataset "thermal600=$DATA/test.xyz" --dataset "harmonic=$HARMONIC" \
    --base-model "$V11" --delta-model "$MODEL" --device cuda \
    --output "$LANE/gate_metrics.json"
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/select_graphene_fd_delta_model.py \
    --metrics "$LANE/gate_metrics.json" --thermal-label thermal600 \
    --output "$LANE/gate_selection.json"
touch "$OUT/W12_DONE"
echo "=== weighted delta32 replayw12 COMPLETE $(date -Is) ==="
