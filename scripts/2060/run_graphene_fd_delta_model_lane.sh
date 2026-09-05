#!/usr/bin/env bash
# Train a conservative additive correction on top of frozen v11.  The targets
# already have the provisional Fermi-Dirac harmonic long-range force removed.
set -euo pipefail

TEMPERATURE="${1:?usage: run_graphene_fd_delta_model_lane.sh 300|600}"
case "$TEMPERATURE" in 300|600) ;; *) exit 2 ;; esac
ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DATA="$ROOT/data/graphene_fd_delta_pilot"
OUT="$ROOT/results/graphene_fd_delta_pilot/T${TEMPERATURE}"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
RAW="$ROOT/results/graphene_fd_model_ablation_v2/T${TEMPERATURE}"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

while [[ ! -e "$RAW/DONE" ]]; do
    echo "waiting for the already-running raw-force diagnostic to release GPU: $(date -Is)"
    sleep 60
done
for path in "$V11" "$DATA/T${TEMPERATURE}/train.xyz" \
    "$DATA/T${TEMPERATURE}/val.xyz" "$DATA/T${TEMPERATURE}/test.xyz" \
    "$DATA/T${TEMPERATURE}/long_range_operator.npz" \
    "$ROOT/data/finetune_graphene8/val.xyz"; do
    [[ -s "$path" ]]
done
[[ ! -e "$OUT/DONE" ]] || exit 0

train_candidate() {
    local variant="$1" irreps="$2" lr="$3" epochs="$4" patience="$5" seed="$6"
    local lane="$OUT/$variant" name="gr_fd${TEMPERATURE}_${variant}"
    local model="$lane/${name}.model"
    mkdir -p "$lane/checkpoints" "$lane/logs" "$lane/results"
    if [[ ! -s "$model" ]]; then
        local restart=()
        if find "$lane/checkpoints" -type f -name '*.pt' -print -quit | grep -q .; then
            restart+=(--restart_latest)
        fi
        echo "=== T=$TEMPERATURE frozen-v11 delta variant=$variant start $(date -Is) ==="
        "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
            --name "$name" --model MACE --num_interactions 2 \
            --hidden_irreps "$irreps" --r_max 5.0 --num_radial_basis 8 \
            --num_cutoff_basis 5 --correlation 2 \
            --train_file "$DATA/T${TEMPERATURE}/train.xyz" \
            --valid_file "$DATA/T${TEMPERATURE}/val.xyz" \
            --test_file "$DATA/T${TEMPERATURE}/test.xyz" \
            --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
            --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
            --batch_size 1 --valid_batch_size 1 --max_num_epochs "$epochs" \
            --patience "$patience" --eval_interval 2 --lr "$lr" \
            --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
            --default_dtype float32 --device cuda --seed "$seed" --save_cpu \
            --model_dir "$lane" --checkpoints_dir "$lane/checkpoints" \
            --log_dir "$lane/logs" --results_dir "$lane/results" \
            "${restart[@]}"
        [[ -s "$model" ]]
    fi

    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/evaluate_graphene_fd_delta_model.py \
        --dataset "thermal${TEMPERATURE}=$DATA/T${TEMPERATURE}/test.xyz" \
        --dataset "harmonic=data/finetune_graphene8/val.xyz" \
        --base-model "$V11" --delta-model "$model" --device cuda \
        --output "$lane/gate_metrics.json"
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/select_graphene_fd_delta_model.py \
        --metrics "$lane/gate_metrics.json" \
        --thermal-label "thermal${TEMPERATURE}" \
        --output "$lane/gate_selection.json"
    local passes
    passes="$($CONDA run -n phonon python -c \
        'import json,sys; print(int(json.load(open(sys.argv[1]))["passes_force_and_replay_gate"]))' \
        "$lane/gate_selection.json" | tail -1)"
    if [[ "$passes" == 1 ]]; then
        cp -p "$lane/gate_selection.json" "$OUT/selection.json"
        cp -p "$lane/gate_metrics.json" "$OUT/selected_gate_metrics.json"
        echo "$variant" > "$OUT/selected_variant.txt"
        touch "$OUT/CANDIDATE_PASSED" "$OUT/DONE"
        echo "=== T=$TEMPERATURE selected frozen-v11 delta variant=$variant $(date -Is) ==="
        return 0
    fi
    echo "T=$TEMPERATURE frozen-v11 delta variant=$variant failed independent gate"
    return 1
}

selected=0
if train_candidate delta32 '32x0e+32x1o' 0.001 240 50 71; then selected=1; fi
if [[ "$selected" == 0 ]] && train_candidate delta16 '16x0e+16x1o' 0.001 260 60 73; then selected=1; fi
if [[ "$selected" == 0 ]] && train_candidate delta32low '32x0e+32x1o' 0.0005 320 70 79; then selected=1; fi
if [[ "$selected" == 0 ]]; then
    touch "$OUT/BLOCKED_FORCE_GATE" "$OUT/DONE"
fi

if [[ "$TEMPERATURE" == 600 ]]; then
    REMOTE="howardwang@100.105.21.7"
    DEST=/home/howardwang/phonon/results/graphene_fd_delta_pilot/T600
    SSH=(env -u LD_LIBRARY_PATH ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
    SCP=(env -u LD_LIBRARY_PATH scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
    "${SSH[@]}" "$REMOTE" "mkdir -p '$DEST'"
    if [[ "$selected" == 1 ]]; then
        variant=$(<"$OUT/selected_variant.txt")
        name="gr_fd600_${variant}.model"
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
        "${SSH[@]}" "$REMOTE" "touch '$DEST/BLOCKED_FORCE_GATE' '$DEST/REMOTE_DONE'"
    fi
fi
echo "=== graphene frozen-v11 delta lane T=$TEMPERATURE COMPLETE $(date -Is) ==="
