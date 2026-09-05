#!/usr/bin/env bash
# Conservative temperature-specific fine-tuning with immediate retention gates.
set -euo pipefail

TEMPERATURE="${1:?usage: run_graphene_fd_model_ablation_v2_lane.sh 300|600}"
case "$TEMPERATURE" in 300|600) ;; *) exit 2 ;; esac
ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DATA="$ROOT/data/graphene_fd_model_ablation_v2/replay72_joint"
OUT="$ROOT/results/graphene_fd_model_ablation_v2/T${TEMPERATURE}"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

for path in "$V11" "$DATA/T${TEMPERATURE}/train.xyz" \
    "$DATA/T${TEMPERATURE}/val.xyz" "$DATA/T${TEMPERATURE}/test.xyz" \
    "$ROOT/data/finetune_graphene8/val.xyz"; do
    [[ -s "$path" ]]
done
[[ ! -e "$OUT/DONE" ]] || exit 0

train_candidate() {
    local variant="$1" lr="$2" epochs="$3" patience="$4" seed="$5"
    shift 5
    local lane="$OUT/$variant" name="gr_fd${TEMPERATURE}_v2_${variant}"
    local model="$lane/${name}.model"
    mkdir -p "$lane/checkpoints" "$lane/logs" "$lane/results"
    if [[ ! -s "$model" ]]; then
        local restart=()
        if find "$lane/checkpoints" -type f -name '*.pt' -print -quit | grep -q .; then
            restart+=(--restart_latest)
        fi
        echo "=== T=$TEMPERATURE v2 variant=$variant start $(date -Is) ==="
        "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
            --name "$name" --foundation_model "$V11" \
            --multiheads_finetuning False --foundation_model_elements True \
            --train_file "$DATA/T${TEMPERATURE}/train.xyz" \
            --valid_file "$DATA/T${TEMPERATURE}/val.xyz" \
            --test_file "$DATA/T${TEMPERATURE}/test.xyz" \
            --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
            --energy_weight 0.0 --forces_weight 100.0 \
            --batch_size 1 --valid_batch_size 1 --max_num_epochs "$epochs" \
            --patience "$patience" --eval_interval 2 --lr "$lr" \
            --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
            --default_dtype float32 --device cuda --seed "$seed" --save_cpu \
            --model_dir "$lane" --checkpoints_dir "$lane/checkpoints" \
            --log_dir "$lane/logs" --results_dir "$lane/results" \
            "$@" "${restart[@]}"
        [[ -s "$model" ]]
    fi

    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/evaluate_graphene_model_variants.py \
        --dataset "thermal${TEMPERATURE}=$DATA/T${TEMPERATURE}/test.xyz" \
        --dataset "harmonic=data/finetune_graphene8/val.xyz" \
        --model "v11=$V11" --model "${variant}_${TEMPERATURE}=$model" \
        --device cuda --output "$lane/gate_metrics.json"
    selection_args=(
        --metrics "$lane/gate_metrics.json"
        --output "$lane/gate_selection.json"
    )
    if [[ "$TEMPERATURE" == 300 ]]; then
        selection_args+=(--candidate-300 "${variant}_300")
    else
        selection_args+=(--candidate-600 "${variant}_600")
    fi
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/select_graphene_model_variants.py \
        "${selection_args[@]}"
    local passes
    passes="$($CONDA run -n phonon python -c \
        'import json,sys; print(int(json.load(open(sys.argv[1]))["passes_all_requested_temperature_gates"]))' \
        "$lane/gate_selection.json" | tail -1)"
    if [[ "$passes" == 1 ]]; then
        cp -p "$lane/gate_selection.json" "$OUT/selection.json"
        cp -p "$lane/gate_metrics.json" "$OUT/selected_gate_metrics.json"
        echo "$variant" > "$OUT/selected_variant.txt"
        touch "$OUT/CANDIDATE_PASSED" "$OUT/DONE"
        echo "=== T=$TEMPERATURE v2 selected variant=$variant $(date -Is) ==="
        return 0
    fi
    echo "T=$TEMPERATURE v2 variant=$variant failed retention gate"
    return 1
}

selected=0
if train_candidate low_lr 0.00002 120 30 53; then selected=1; fi
if [[ "$selected" == 0 ]] && train_candidate lora4 0.0001 160 40 59 \
    --lora True --lora_rank 4 --lora_alpha 1.0; then selected=1; fi
if [[ "$selected" == 0 ]] && train_candidate freeze5 0.00005 160 40 61 \
    --freeze 5; then selected=1; fi
if [[ "$selected" == 0 ]]; then
    touch "$OUT/BLOCKED_FORCE_GATE" "$OUT/DONE"
fi

if [[ "$TEMPERATURE" == 600 ]]; then
    REMOTE="howardwang@100.105.21.7"
    DEST=/home/howardwang/phonon/results/graphene_fd_model_ablation_v2/T600
    SSH=(env -u LD_LIBRARY_PATH ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
    SCP=(env -u LD_LIBRARY_PATH scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
    "${SSH[@]}" "$REMOTE" "mkdir -p '$DEST'"
    if [[ "$selected" == 1 ]]; then
        variant=$(<"$OUT/selected_variant.txt")
        name="gr_fd600_v2_${variant}.model"
        "${SSH[@]}" "$REMOTE" "mkdir -p '$DEST/$variant'"
        for source in "$OUT/$variant/$name" "$OUT/selection.json" \
            "$OUT/selected_gate_metrics.json" "$OUT/selected_variant.txt" "$LOG"; do
            filename=$(basename "$source")
            target="$DEST/$filename"
            [[ "$source" == "$OUT/$variant/$name" ]] && target="$DEST/$variant/$name"
            "${SCP[@]}" "$source" "$REMOTE:$target.partial"
            "${SSH[@]}" "$REMOTE" "mv '$target.partial' '$target'"
        done
        "${SSH[@]}" "$REMOTE" "touch '$DEST/REMOTE_DONE'"
    else
        "${SCP[@]}" "$LOG" "$REMOTE:$DEST/run.log.partial"
        "${SSH[@]}" "$REMOTE" "mv '$DEST/run.log.partial' '$DEST/run.log'"
        "${SSH[@]}" "$REMOTE" "touch '$DEST/BLOCKED_FORCE_GATE'"
    fi
fi
echo "=== graphene model ablation v2 T=$TEMPERATURE COMPLETE $(date -Is) ==="
