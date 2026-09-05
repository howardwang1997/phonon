#!/usr/bin/env bash
# Train one temperature lane for the fixed-data replay/continuation ablation.
set -euo pipefail

TEMPERATURE="${1:?usage: run_graphene_fd_model_ablation_lane.sh 300|600}"
case "$TEMPERATURE" in 300|600) ;; *) exit 2 ;; esac
ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DATA="$ROOT/data/graphene_fd_model_ablation"
OUT="$ROOT/results/graphene_fd_model_ablation/T${TEMPERATURE}"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
WAVE2="$ROOT/results/graphene_fd_thermal_finetune_wave2/T${TEMPERATURE}/gr_v11_fd${TEMPERATURE}_wave2.model"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

for path in "$V11" "$WAVE2" \
    "$DATA/replay12/T${TEMPERATURE}/train.xyz" \
    "$DATA/replay0/T${TEMPERATURE}/train.xyz" \
    "$DATA/replay0/T${TEMPERATURE}/val.xyz" \
    "$DATA/replay0/T${TEMPERATURE}/test.xyz"; do
    [[ -s "$path" ]]
done

train_variant() {
    local variant="$1" foundation="$2" dataset="$3" lr="$4"
    local epochs="$5" patience="$6" seed="$7"
    local name="gr_fd${TEMPERATURE}_${variant}" lane="$OUT/$variant"
    local model="$lane/${name}.model"
    mkdir -p "$lane/checkpoints" "$lane/logs" "$lane/results"
    if [[ -s "$model" ]]; then
        echo "T=$TEMPERATURE variant=$variant already complete"
        return 0
    fi
    local restart=()
    if find "$lane/checkpoints" -type f -name '*.pt' -print -quit | grep -q .; then
        restart+=(--restart_latest)
    fi
    echo "=== T=$TEMPERATURE variant=$variant start $(date -Is) ==="
    "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
        --name "$name" --foundation_model "$foundation" \
        --multiheads_finetuning False --foundation_model_elements True \
        --train_file "$DATA/$dataset/T${TEMPERATURE}/train.xyz" \
        --valid_file "$DATA/$dataset/T${TEMPERATURE}/val.xyz" \
        --test_file "$DATA/$dataset/T${TEMPERATURE}/test.xyz" \
        --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
        --energy_weight 0.0 --forces_weight 100.0 \
        --batch_size 1 --valid_batch_size 1 --max_num_epochs "$epochs" \
        --patience "$patience" --eval_interval 2 --lr "$lr" \
        --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
        --default_dtype float32 --device cuda --seed "$seed" --save_cpu \
        --model_dir "$lane" --checkpoints_dir "$lane/checkpoints" \
        --log_dir "$lane/logs" --results_dir "$lane/results" \
        "${restart[@]}"
    [[ -s "$model" ]]
    echo "=== T=$TEMPERATURE variant=$variant complete $(date -Is) ==="
}

train_variant replay12 "$V11" replay12 0.0002 320 60 41
train_variant thermal_only "$V11" replay0 0.0001 320 60 43
train_variant continued "$WAVE2" replay0 0.00005 180 40 47
touch "$OUT/DONE"

if [[ "$TEMPERATURE" == 600 ]]; then
    REMOTE="howardwang@100.105.21.7"
    DEST=/home/howardwang/phonon/results/graphene_fd_model_ablation/T600
    SSH=(env -u LD_LIBRARY_PATH ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
    SCP=(env -u LD_LIBRARY_PATH scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
    "${SSH[@]}" "$REMOTE" "mkdir -p '$DEST/replay12' '$DEST/thermal_only' '$DEST/continued'"
    for variant in replay12 thermal_only continued; do
        name="gr_fd600_${variant}.model"
        "${SCP[@]}" "$OUT/$variant/$name" "$REMOTE:$DEST/$variant/$name.partial"
        "${SSH[@]}" "$REMOTE" \
            "mv '$DEST/$variant/$name.partial' '$DEST/$variant/$name'"
    done
    "${SCP[@]}" "$LOG" "$REMOTE:$DEST/run.log.partial"
    "${SSH[@]}" "$REMOTE" \
        "mv '$DEST/run.log.partial' '$DEST/run.log'; touch '$DEST/REMOTE_DONE'"
fi
echo "=== graphene model ablation T=$TEMPERATURE COMPLETE $(date -Is) ==="
