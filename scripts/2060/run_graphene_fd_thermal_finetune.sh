#!/usr/bin/env bash
# Fine-tune the same v11 backbone separately at the two physical-FD settings,
# evaluate on untouched structures, then run the matching short-range TDEP
# controls.  The .READY marker is written only after both V100 label files have
# been transferred and validated by the local monitor.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
PY_ENV="$HOME/miniconda3/envs/phonon/bin"
SOURCE="$ROOT/data/graphene_fd_thermal_labels"
DATA="$ROOT/data/graphene_fd_thermal_finetune"
OUT="$ROOT/results/graphene_fd_thermal_finetune"
DONE="$OUT/DONE"
LOG="$OUT/run.log"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"

cd "$ROOT"
mkdir -p "$SOURCE" "$DATA" "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "=== graphene physical-FD thermal fine-tune queue start $(date -Is) ==="
while [[ ! -e "$SOURCE/.READY" ]]; do
    echo "waiting for both validated V100 thermal-label files: $(date -Is)"
    sleep 60
done
[[ -s "$SOURCE/A.xyz" && -s "$SOURCE/B.xyz" && -s "$V11" ]]
# The seed sweep is deliberately lower priority and checkpointed.  Stop it as
# soon as the target physical-FD data arrive so the formal fine-tune owns GPU.
systemctl --user stop phonon-graphene-v11-control-seeds.service 2>/dev/null || true
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/prepare_graphene_fd_thermal_finetune.py \
    --labels-a "$SOURCE/A.xyz" --labels-b "$SOURCE/B.xyz" \
    --replay data/finetune_graphene8/train.xyz --replay-count 12 \
    --output "$DATA"

train_temperature() {
    local temperature="$1" name="gr_v11_fd${1}" lane_out="$OUT/T${1}"
    local model="$lane_out/${name}.model"
    mkdir -p "$lane_out/checkpoints" "$lane_out/logs" "$lane_out/results"
    if [[ -s "$model" ]]; then
        echo "T=$temperature model already complete: $model"
        return 0
    fi
    local restart=()
    if find "$lane_out/checkpoints" -type f -name '*.pt' -print -quit | grep -q .; then
        restart+=(--restart_latest)
    fi
    echo "--- T=$temperature v11 physical-FD fine-tune $(date -Is) ---"
    "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
        --name "$name" \
        --foundation_model "$V11" \
        --multiheads_finetuning False --foundation_model_elements True \
        --train_file "$DATA/T${temperature}/train.xyz" \
        --valid_file "$DATA/T${temperature}/val.xyz" \
        --test_file "$DATA/T${temperature}/val.xyz" \
        --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
        --energy_weight 0.0 --forces_weight 100.0 \
        --batch_size 1 --valid_batch_size 1 \
        --max_num_epochs 200 --patience 40 --eval_interval 2 \
        --lr 0.0005 --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
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
    --model "fd300=$OUT/T300/gr_v11_fd300.model" \
    --model "fd600=$OUT/T600/gr_v11_fd600.model" \
    --device cuda --output "$OUT/force_validation.json"

run_tdep() {
    local temperature="$1" tag="graphene_v11_fd${1}_short_range"
    local model="$OUT/T${1}/gr_v11_fd${1}.model"
    if [[ -s "results/td_phonon/td_${tag}.npz" && -s "results/td_phonon/td_${tag}.csv" ]]; then
        echo "T=$temperature short-range TDEP already complete"
        return 0
    fi
    "$CONDA" run --no-capture-output -n phonon python scripts/td_phonon.py \
        --structure data/td_phonon/graphene.xyz --model "$model" --device cuda \
        --tag "$tag" --outdir results/td_phonon --temperatures "$temperature" \
        --supercell 6,6,1 --no-relax --a 2.4600000087 \
        --equil 1500 --nsnap 120 --stride 40 --npoints 201 \
        --checkpoint-root "results/td_phonon/${tag}_checkpoint" \
        --checkpoint-every 25
}

run_tdep 300
run_tdep 600
touch "$DONE"
echo "=== graphene physical-FD thermal fine-tune queue COMPLETE $(date -Is) ==="
