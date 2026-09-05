#!/usr/bin/env bash
# Controlled 600 K delta-MACE sweep with an explicit harmonic-replay weight.
# All pre-existing pilot artifacts remain untouched.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DATA="$ROOT/data/graphene_fd_delta_pilot/T600"
OUT="$ROOT/results/graphene_fd_delta_weighted/T600"
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

for path in "$V11" "$DATA/train.xyz" "$DATA/val.xyz" "$DATA/test.xyz" "$HARMONIC"; do
    [[ -s "$path" ]]
done
if [[ -e "$OUT/DONE" ]]; then
    echo "weighted 600 K delta sweep already complete"
    exit 0
fi

train_weight() {
    local weight="$1"
    local tag="replayw${weight}"
    local lane="$OUT/$tag"
    local name="gr_fd600_delta32_${tag}"
    local model="$lane/${name}.model"
    mkdir -p "$lane/checkpoints" "$lane/logs" "$lane/results"

    if [[ ! -s "$model" ]]; then
        echo "=== weighted delta32 tag=$tag start $(date -Is) ==="
        "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
            --name "$name" --model MACE --num_interactions 2 \
            --hidden_irreps '32x0e+32x1o' --r_max 5.0 --num_radial_basis 8 \
            --num_cutoff_basis 5 --correlation 2 \
            --train_file "$DATA/train.xyz" --valid_file "$DATA/val.xyz" \
            --test_file "$DATA/test.xyz" \
            --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
            --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
            --config_type_weights "{'physical_fd_thermal_600K':1.0,'v11_fc_distillation_replay':${weight}.0,'physical_fd_thermal_600K_validation':1.0,'Default':${weight}.0}" \
            --batch_size 1 --valid_batch_size 1 --max_num_epochs 180 \
            --patience 35 --eval_interval 5 --lr 0.001 \
            --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
            --default_dtype float32 --device cuda --seed 83 --save_cpu \
            --keep_checkpoints --save_all_checkpoints \
            --model_dir "$lane" --checkpoints_dir "$lane/checkpoints" \
            --log_dir "$lane/logs" --results_dir "$lane/results"
    fi
    [[ -s "$model" ]]

    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/evaluate_graphene_fd_delta_model.py \
        --dataset "thermal600=$DATA/test.xyz" \
        --dataset "harmonic=$HARMONIC" \
        --base-model "$V11" --delta-model "$model" --device cuda \
        --output "$lane/gate_metrics.json"
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/select_graphene_fd_delta_model.py \
        --metrics "$lane/gate_metrics.json" --thermal-label thermal600 \
        --output "$lane/gate_selection.json"
    echo "=== weighted delta32 tag=$tag complete $(date -Is) ==="
}

for weight in 4 8 16; do
    train_weight "$weight"
done

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/summarize_graphene_fd_delta_weight_sweep.py \
    --candidate "replayw4=$OUT/replayw4/gate_selection.json" \
    --candidate "replayw8=$OUT/replayw8/gate_selection.json" \
    --candidate "replayw16=$OUT/replayw16/gate_selection.json" \
    --output "$OUT/weight_sweep_summary.json"

selected="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected_tag"])' \
    "$OUT/weight_sweep_summary.json" | tail -1)"
printf '%s\n' "$selected" > "$OUT/selected_tag.txt"
if "$CONDA" run -n phonon python -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["status"] == "development_candidate_found" else 1)' \
    "$OUT/weight_sweep_summary.json"; then
    touch "$OUT/CANDIDATE_PASSED_DEVELOPMENT_GATE"
else
    touch "$OUT/BLOCKED_DEVELOPMENT_GATE"
fi
touch "$OUT/DONE"
echo "=== weighted 600 K delta sweep COMPLETE $(date -Is) ==="
