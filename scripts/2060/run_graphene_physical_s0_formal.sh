#!/usr/bin/env bash
# Formal 240-epoch S0 development run on the released 300/450/600 K dataset.
# REPLAY_WEIGHT=12 is the baseline; the only allowed controlled ablation is 16.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DATA="$ROOT/data/graphene_physical_s0"
MANIFEST="$DATA/manifest.json"
BASE="$ROOT/results/gr_backbone_v11/ft_graphene.model"
REPLAY_WEIGHT="${REPLAY_WEIGHT:-12}"
case "$REPLAY_WEIGHT" in 12|16) ;; *) echo "REPLAY_WEIGHT must be 12 or 16" >&2; exit 2 ;; esac
OUT="${OUT:-$ROOT/results/graphene_physics_temperature/post_p4_feasibility/S0_unified_short/formal_240ep_2060_seed83}"
NAME="${NAME:-gr_physical_s0_delta32_seed83}"
MODEL="$OUT/$NAME.model"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT/checkpoints" "$OUT/logs" "$OUT/results"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$OUT/DONE" ]]; then
    echo "S0 formal seed83 experiment already complete"
    exit 0
fi
for path in "$MANIFEST" "$BASE" "$DATA/train.xyz" "$DATA/val.xyz" \
    "$DATA/test.xyz" "$DATA/T300/test.xyz" "$DATA/T450/test.xyz" \
    "$DATA/T600/test.xyz" "$DATA/replay/val.xyz"; do
    [[ -s "$path" ]]
done

"$CONDA" run -n phonon python -c '
import hashlib, json, pathlib, sys
manifest_path, base_path = map(pathlib.Path, sys.argv[1:])
m = json.loads(manifest_path.read_text())
h = hashlib.sha256(base_path.read_bytes()).hexdigest()
assert m["status"] == "passed"
assert all(m["aggregate_gates"]["checks"].values())
assert m["development_temperatures_K"] == [300, 450, 600]
assert m["locked_validation_temperatures_K_not_accessed"] == [375, 525]
assert m["temperature_or_degauss_is_model_input"] is False
assert h == m["base_model_sha256"]
' "$MANIFEST" "$BASE"

CONFIG_WEIGHTS="$($CONDA run -n phonon python -c '
import json, sys
m = json.load(open(sys.argv[1]))
weights = dict(m["recommended_config_weights"]["config_type_weights"])
replay_weight = float(sys.argv[2])
weights["Default"] = replay_weight
weights["physical_s0_harmonic_replay"] = replay_weight
weights["physical_s0_harmonic_replay_validation"] = replay_weight
print(repr(weights))
' "$MANIFEST" "$REPLAY_WEIGHT" | tail -1)"

S0_GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( S0_GPU_USED_MIB > 1000 )); then
    echo "GPU is no longer idle: ${S0_GPU_USED_MIB} MiB already used"
    exit 2
fi

S0_START_EPOCH="$(date +%s)"
date -Is > "$OUT/STARTED_AT"
touch "$OUT/RUNNING"
echo "timestamp_iso,gpu_util_percent,memory_used_MiB,memory_total_MiB" > "$OUT/gpu_samples.csv"
(
    while [[ -e "$OUT/RUNNING" ]]; do
        S0_GPU_SAMPLE="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)"
        echo "$(date -Is),$S0_GPU_SAMPLE" >> "$OUT/gpu_samples.csv"
        sleep 10
    done
) &
S0_MONITOR_PID=$!

finish() {
    S0_EXIT_CODE=$?
    trap - EXIT
    rm -f "$OUT/RUNNING"
    kill "$S0_MONITOR_PID" 2>/dev/null || true
    wait "$S0_MONITOR_PID" 2>/dev/null || true
    echo "$S0_EXIT_CODE" > "$OUT/EXIT_CODE"
    if (( S0_EXIT_CODE != 0 )); then
        touch "$OUT/FAILED"
    fi
    exit "$S0_EXIT_CODE"
}
trap finish EXIT

echo "=== S0 formal seed83 replay_weight=$REPLAY_WEIGHT START $(date -Is), initial GPU ${S0_GPU_USED_MIB} MiB ==="
echo "config weights: $CONFIG_WEIGHTS"
S0_RESTART_ARGS=()
if compgen -G "$OUT/checkpoints/${NAME}_run-*_epoch-*.pt" >/dev/null; then
    S0_RESTART_ARGS+=(--restart_latest)
    echo "resuming from the latest complete MACE checkpoint"
fi
"$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
    --name "$NAME" --model MACE --num_interactions 2 \
    --hidden_irreps '32x0e+32x1o' --r_max 5.0 --num_radial_basis 8 \
    --num_cutoff_basis 5 --correlation 2 \
    --train_file "$DATA/train.xyz" --valid_file "$DATA/val.xyz" \
    --test_file "$DATA/test.xyz" \
    --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
    --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
    --config_type_weights "$CONFIG_WEIGHTS" \
    --batch_size 1 --valid_batch_size 1 --max_num_epochs 240 \
    --patience 60 --eval_interval 5 --lr 0.001 \
    --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
    --default_dtype float32 --device cuda --seed 83 --save_cpu \
    --keep_checkpoints --save_all_checkpoints \
    --model_dir "$OUT" --checkpoints_dir "$OUT/checkpoints" \
    --log_dir "$OUT/logs" --results_dir "$OUT/results" \
    "${S0_RESTART_ARGS[@]}"

[[ -s "$MODEL" ]]
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_fd_joint_checkpoints.py \
    --base-model "$BASE" --template-model "$MODEL" \
    --checkpoint-dir "$OUT/checkpoints" \
    --thermal "thermal300=$DATA/T300/test.xyz" \
    --thermal "thermal450=$DATA/T450/test.xyz" \
    --thermal "thermal600=$DATA/T600/test.xyz" \
    --harmonic "$DATA/replay/val.xyz" --device cuda \
    --force-rmse-threshold 50 --force-max-threshold 250 \
    --harmonic-rmse-ratio 2 \
    --scope "S0 replay-weight-$REPLAY_WEIGHT 300/450/600 K development checkpoint selection; 375/525 K remain locked" \
    --output "$OUT/checkpoint_sweep.json" \
    --selected-model "$OUT/selected_checkpoint.model"

S0_END_EPOCH="$(date +%s)"
"$CONDA" run -n phonon python -c '
import json, pathlib, sys
start, end = map(int, sys.argv[1:3])
sweep = json.load(open(sys.argv[3]))
path = pathlib.Path(sys.argv[4])
path.write_text(json.dumps({
    "status": "complete",
    "wall_time_seconds_including_evaluation": end - start,
    "training_epochs_requested": 240,
    "harmonic_replay_weight": float(sys.argv[5]),
    "device": "RTX 2060 SUPER",
    "checkpoint_gate_status": sweep["status"],
    "selected_epoch": sweep["selected"]["epoch"],
}, indent=2) + "\n")
' "$S0_START_EPOCH" "$S0_END_EPOCH" "$OUT/checkpoint_sweep.json" "$OUT/runtime.json" "$REPLAY_WEIGHT"

S0_GATE_STATUS="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$OUT/checkpoint_sweep.json" | tail -1)"
if [[ "$S0_GATE_STATUS" == "checkpoint_passed" ]]; then
    touch "$OUT/PASSED_DEVELOPMENT_FORCE_GATE"
else
    touch "$OUT/NO_CHECKPOINT_PASSED_DEVELOPMENT_FORCE_GATE"
fi
date -Is > "$OUT/COMPLETED_AT"
touch "$OUT/DONE"
echo "=== S0 formal seed83 replay_weight=$REPLAY_WEIGHT COMPLETE status=$S0_GATE_STATUS $(date -Is) ==="
