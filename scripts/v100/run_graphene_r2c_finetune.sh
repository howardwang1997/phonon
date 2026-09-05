#!/usr/bin/env bash
# Low-learning-rate fine-tuning of the frozen current S0 short-range model.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon}"
DATA="${DATA:-/data/graphene_r2c_loss_only}"
INIT_MODEL="${INIT_MODEL:-/data/graphene_r2c_loss_only/current_s0_selected.model}"
SEED="${SEED:-83}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
NEW_CONFIG_WEIGHT="${NEW_CONFIG_WEIGHT:-}"
REPLAY_WEIGHT="${REPLAY_WEIGHT:-}"
RUN_TAG="${RUN_TAG:-finetune_forces}"
OUT="${OUT:-/data/graphene_r2c_short_repair/finetune_forces_only_seed${SEED}}"
NAME="gr_r2c_${RUN_TAG}_seed${SEED}"
MODEL="$OUT/$NAME.model"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT/checkpoints" "$OUT/logs" "$OUT/results"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

if [[ -e "$OUT/TRAINING_DONE" ]]; then
    echo "R2C fine-tune seed=$SEED already complete"
    exit 0
fi
for path in "$DATA/manifest.json" "$DATA/train.xyz" "$DATA/val.xyz" \
    "$DATA/test.xyz" "$DATA/supported9_train.xyz" "$INIT_MODEL"; do
    [[ -s "$path" ]]
done

CONFIG_WEIGHTS="$($CONDA run -n "$CONDA_ENV" python -c '
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert manifest["status"] == "frozen_before_R2C_training"
assert manifest["long_range_model_modified"] is False
assert manifest["new_DFT"]["n_supported"] == 9
weights = dict(manifest["config_type_weights"])
new_weight, replay_weight = sys.argv[2:]
if new_weight:
    weights["r2c_fixed_smearing_supported"] = float(new_weight)
if replay_weight:
    weights["Default"] = float(replay_weight)
    weights["physical_s0_harmonic_replay"] = float(replay_weight)
    weights["physical_s0_harmonic_replay_validation"] = float(replay_weight)
print(repr(weights))
' "$DATA/manifest.json" "$NEW_CONFIG_WEIGHT" "$REPLAY_WEIGHT" | tail -1)"

GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( GPU_USED_MIB > 1000 )); then
    echo "GPU is not idle: ${GPU_USED_MIB} MiB already used" >&2
    exit 2
fi

"$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib, json, pathlib, sys
data, initial, output, seed, epochs, lr, conda_env, new_weight, replay_weight, run_tag, script = sys.argv[1:]
data = pathlib.Path(data); initial = pathlib.Path(initial)
output = pathlib.Path(output); script = pathlib.Path(script)
digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
payload = {
    "status": "frozen_before_training",
    "mode": "current_S0_low_lr_forces_only_finetune",
    "seed": int(seed), "max_epochs": int(epochs), "learning_rate": float(lr),
    "conda_environment": conda_env,
    "run_tag": run_tag,
    "config_weight_overrides": {
        "r2c_fixed_smearing_supported": (float(new_weight) if new_weight else None),
        "harmonic_replay": (float(replay_weight) if replay_weight else None),
    },
    "initial_model": {"path": str(initial), "sha256": digest(initial)},
    "data_manifest": {"path": str(data / "manifest.json"),
                      "sha256": digest(data / "manifest.json")},
    "training_script": {"path": str(script), "sha256": digest(script)},
    "loss": {"name": "forces_only", "energy_weight": 0.0,
             "forces_weight": 100.0},
    "architecture_change": False,
    "long_range_model_modified": False,
}
output.mkdir(parents=True, exist_ok=True)
(output / "training_freeze.json").write_text(json.dumps(payload, indent=2) + "\n")
' "$DATA" "$INIT_MODEL" "$OUT" "$SEED" "$MAX_EPOCHS" "$LEARNING_RATE" "$CONDA_ENV" \
    "$NEW_CONFIG_WEIGHT" "$REPLAY_WEIGHT" "$RUN_TAG" \
    "$ROOT/scripts/v100/run_graphene_r2c_finetune.sh"

START_EPOCH="$(date +%s)"
date -Is > "$OUT/STARTED_AT"
touch "$OUT/RUNNING"
echo "timestamp_iso,gpu_util_percent,memory_used_MiB,memory_total_MiB" > "$OUT/gpu_samples.csv"
(
    while [[ -e "$OUT/RUNNING" ]]; do
        GPU_SAMPLE="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)"
        echo "$(date -Is),$GPU_SAMPLE" >> "$OUT/gpu_samples.csv"
        sleep 10
    done
) &
MONITOR_PID=$!

finish() {
    EXIT_STATUS=$?
    trap - EXIT
    rm -f "$OUT/RUNNING"
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    echo "$EXIT_STATUS" > "$OUT/EXIT_CODE"
    if (( EXIT_STATUS != 0 )); then touch "$OUT/FAILED"; fi
    exit "$EXIT_STATUS"
}
trap finish EXIT

echo "=== R2C fine-tune seed=$SEED START $(date -Is), initial GPU ${GPU_USED_MIB} MiB ==="
echo "initial model: $INIT_MODEL"
echo "config weights: $CONFIG_WEIGHTS"
RESTART_ARGS=()
if compgen -G "$OUT/checkpoints/${NAME}_run-*_epoch-*.pt" >/dev/null; then
    RESTART_ARGS+=(--restart_latest)
fi

"$CONDA" run --no-capture-output -n "$CONDA_ENV" python -m mace.cli.run_train \
    --name "$NAME" --foundation_model "$INIT_MODEL" \
    --multiheads_finetuning False --E0s '{6:0.0}' \
    --train_file "$DATA/train.xyz" --valid_file "$DATA/val.xyz" \
    --test_file "$DATA/test.xyz" \
    --energy_key REF_energy --forces_key REF_forces \
    --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
    --config_type_weights "$CONFIG_WEIGHTS" \
    --batch_size 1 --valid_batch_size 1 --max_num_epochs "$MAX_EPOCHS" \
    --patience 30 --eval_interval 5 --lr "$LEARNING_RATE" \
    --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
    --default_dtype float32 --device cuda --seed "$SEED" --save_cpu \
    --keep_checkpoints --save_all_checkpoints \
    --model_dir "$OUT" --checkpoints_dir "$OUT/checkpoints" \
    --log_dir "$OUT/logs" --results_dir "$OUT/results" \
    "${RESTART_ARGS[@]}"

[[ -s "$MODEL" ]]
END_EPOCH="$(date +%s)"
"$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib, json, pathlib, sys
start, end, model, output = sys.argv[1:]
model = pathlib.Path(model); output = pathlib.Path(output)
(output / "training_runtime.json").write_text(json.dumps({
    "status": "training_complete_pending_checkpoint_gate",
    "wall_time_seconds": int(end) - int(start),
    "final_model": str(model),
    "final_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
}, indent=2) + "\n")
' "$START_EPOCH" "$END_EPOCH" "$MODEL" "$OUT"
date -Is > "$OUT/TRAINING_COMPLETED_AT"
touch "$OUT/TRAINING_DONE"
echo "=== R2C fine-tune seed=$SEED TRAINING COMPLETE $(date -Is) ==="
