#!/usr/bin/env bash
# Train and select one joint 300/600 K short-range delta model before 450 K labels.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
SOURCE="$ROOT/data/graphene_fd_delta_pilot"
SOURCE_MANIFEST="$SOURCE/manifest.json"
DATA="$ROOT/data/graphene_fd_transferability/joint_short"
OUT="$ROOT/results/graphene_fd_transferability/P2_joint_short"
P1="$ROOT/results/graphene_fd_transferability/P1_cross_temperature/cross_temperature_force_matrix.json"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
HARMONIC="$ROOT/data/finetune_graphene8/val.xyz"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$OUT/DONE" ]]; then
    echo "P2 joint short-model experiment already complete"
    exit 0
fi
for path in "$P1" "$SOURCE_MANIFEST" "$V11" "$HARMONIC" \
    "$SOURCE/T300/train.xyz" "$SOURCE/T300/val.xyz" "$SOURCE/T300/test.xyz" \
    "$SOURCE/T600/train.xyz" "$SOURCE/T600/val.xyz" "$SOURCE/T600/test.xyz"; do
    [[ -s "$path" ]]
done
"$CONDA" run -n phonon python -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["requires_joint_model"] else 1)' \
    "$P1"

if [[ ! -s "$DATA/manifest.json" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/prepare_graphene_fd_joint_delta_data.py \
        --input-root "$SOURCE" --source-manifest "$SOURCE_MANIFEST" --output "$DATA"
fi
T300_WEIGHT="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["recommended_config_weights"]["thermal_300_relative_to_600"])' \
    "$DATA/manifest.json" | tail -1)"
echo "joint thermal config weights: T300=$T300_WEIGHT T600=1.0"

train_and_sweep() {
    local replay_weight="$1"
    local tag="replayw${replay_weight}"
    local lane="$OUT/$tag"
    local name="gr_fd_joint300_600_delta32_${tag}"
    local model="$lane/${name}.model"
    local sweep="$lane/checkpoint_sweep.json"
    mkdir -p "$lane/checkpoints" "$lane/logs" "$lane/results"
    if [[ ! -s "$model" ]]; then
        local restart=()
        if compgen -G "$lane/checkpoints/${name}_run-*_epoch-*.pt" >/dev/null; then
            restart+=(--restart_latest)
        fi
        echo "=== P2 joint $tag training START $(date -Is) ==="
        "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
            --name "$name" --model MACE --num_interactions 2 \
            --hidden_irreps '32x0e+32x1o' --r_max 5.0 --num_radial_basis 8 \
            --num_cutoff_basis 5 --correlation 2 \
            --train_file "$DATA/train.xyz" --valid_file "$DATA/val.xyz" \
            --test_file "$DATA/test.xyz" \
            --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
            --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
            --config_type_weights "{'physical_fd_thermal_300K':${T300_WEIGHT},'physical_fd_thermal_600K':1.0,'physical_fd_thermal_300K_validation':${T300_WEIGHT},'physical_fd_thermal_600K_validation':1.0,'physical_fd_thermal_300K_test':${T300_WEIGHT},'physical_fd_thermal_600K_test':1.0,'v11_fc_distillation_replay':${replay_weight}.0,'v11_fc_distillation_replay_validation':${replay_weight}.0,'Default':${replay_weight}.0}" \
            --batch_size 1 --valid_batch_size 1 --max_num_epochs 180 \
            --patience 35 --eval_interval 5 --lr 0.001 \
            --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
            --default_dtype float32 --device cuda --seed 83 --save_cpu \
            --keep_checkpoints --save_all_checkpoints \
            --model_dir "$lane" --checkpoints_dir "$lane/checkpoints" \
            --log_dir "$lane/logs" --results_dir "$lane/results" \
            "${restart[@]}"
    fi
    [[ -s "$model" ]]
    if [[ ! -s "$sweep" ]]; then
        "$CONDA" run --no-capture-output -n phonon python \
            scripts/smearing_kink/evaluate_graphene_fd_joint_checkpoints.py \
            --base-model "$V11" --template-model "$model" \
            --checkpoint-dir "$lane/checkpoints" \
            --thermal "thermal300=$SOURCE/T300/test.xyz" \
            --thermal "thermal600=$SOURCE/T600/test.xyz" \
            --harmonic "$HARMONIC" --device cuda \
            --scope "300/600 K endpoint development selection before any 450 K labels" \
            --output "$sweep" --selected-model "$lane/selected_checkpoint.model"
    fi
}

echo "=== graphene transferability P2 START $(date -Is) ==="
train_and_sweep 12
status12="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$OUT/replayw12/checkpoint_sweep.json" | tail -1)"
candidates=(--candidate "replayw12=$OUT/replayw12/checkpoint_sweep.json")
if [[ "$status12" != checkpoint_passed ]]; then
    train_and_sweep 16
    candidates+=(--candidate "replayw16=$OUT/replayw16/checkpoint_sweep.json")
fi
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/select_graphene_fd_joint_sweeps.py \
    "${candidates[@]}" --output "$OUT/checkpoint_selection.json"
selection_status="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$OUT/checkpoint_selection.json" | tail -1)"
selected_model="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["selected_model"])' \
    "$OUT/checkpoint_selection.json" | tail -1)"
if [[ "$selection_status" == joint_endpoint_candidate_found ]]; then
    cp -p "$selected_model" "$OUT/selected_checkpoint.model"
    touch "$OUT/PASSED_ENDPOINT_DEVELOPMENT_GATE"
else
    cp -p "$selected_model" "$OUT/provisional_selected_checkpoint.model"
    touch "$OUT/BLOCKED_ENDPOINT_DEVELOPMENT_GATE"
fi
touch "$OUT/DONE"
echo "=== graphene transferability P2 COMPLETE status=$selection_status $(date -Is) ==="
