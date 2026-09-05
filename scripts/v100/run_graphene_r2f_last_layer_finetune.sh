#!/usr/bin/env bash
# Fine-tune only the terminal depth-3 block on support+harmonic full targets.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon}"
DATA="${DATA:-/data/graphene_r2f_joint_support_harmonic}"
INIT_MODEL="${INIT_MODEL:-/data/graphene_r2c_short_repair/depth3_forces_only_seed83/gr_r2c_depth3_forces_only_seed83.model}"
SEED="${SEED:-83}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
RUN_TAG="${RUN_TAG:-depth3_lastblock_direct}"
OUT="${OUT:-/data/graphene_r2f_last_layer/${RUN_TAG}_seed${SEED}}"
NAME="gr_r2f_${RUN_TAG}_seed${SEED}"
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
    echo "R2F last-block seed=$SEED already complete"
    exit 0
fi
for path in "$DATA/manifest.json" "$DATA/train.xyz" "$DATA/val.xyz" \
    "$DATA/test.xyz" "$DATA/supported9_train.xyz" "$INIT_MODEL"; do
    [[ -s "$path" ]]
done

CONFIG_WEIGHTS="$($CONDA run -n "$CONDA_ENV" python -c '
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert manifest["status"] == "frozen_before_R2F_depth3_last_layer_training"
assert manifest["long_range_model_modified"] is False
assert manifest["new_DFT_labels_for_this_stage"] == 0
assert manifest["thermal_structures_used_for_gradient_updates"] == 0
assert manifest["counts"] == {"train": 81, "val": 34, "test": 26, "supported9_train": 9}
print(repr(manifest["config_type_weights"]))
' "$DATA/manifest.json" | tail -1)"

GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( GPU_USED_MIB > 1000 )); then
    echo "GPU is not idle: ${GPU_USED_MIB} MiB already used" >&2
    exit 2
fi

"$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib, json, pathlib, sys
data, initial, output, run_tag, seed, epochs, lr, conda_env, script, trainer = sys.argv[1:]
data = pathlib.Path(data); initial = pathlib.Path(initial)
output = pathlib.Path(output); script = pathlib.Path(script); trainer = pathlib.Path(trainer)
digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
manifest = json.loads((data / "manifest.json").read_text())
payload = {
    "status": "frozen_before_training",
    "mode": "depth3_terminal_interaction_product_readout_direct_finetune",
    "run_tag": run_tag,
    "seed": int(seed),
    "max_epochs": int(epochs),
    "learning_rate": float(lr),
    "conda_environment": conda_env,
    "initial_model": {"path": str(initial), "sha256": digest(initial)},
    "data_manifest": {"path": str(data / "manifest.json"),
                      "sha256": digest(data / "manifest.json")},
    "data_outputs": manifest["outputs"],
    "training_script": {"path": str(script), "sha256": digest(script)},
    "direct_trainer": {"path": str(trainer), "sha256": digest(trainer)},
    "loss": {"name": "forces_only", "energy_weight": 0.0,
             "forces_weight": 100.0},
    "optimizer": {"lr": float(lr), "weight_decay": 1.0e-8,
                  "ema": True, "ema_decay": 0.99},
    "trainable_scope": ["interactions.2.*", "products.2.*", "readouts.2.*"],
    "new_DFT_labels": 0,
    "long_range_model_modified": False,
}
output.mkdir(parents=True, exist_ok=True)
(output / "training_freeze.json").write_text(json.dumps(payload, indent=2) + "\n")
' "$DATA" "$INIT_MODEL" "$OUT" "$RUN_TAG" "$SEED" "$MAX_EPOCHS" \
    "$LEARNING_RATE" "$CONDA_ENV" \
    "$ROOT/scripts/v100/run_graphene_r2f_last_layer_finetune.sh" \
    "$ROOT/scripts/smearing_kink/train_graphene_r2f_last_block.py"

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

echo "=== R2F last-block seed=$SEED START $(date -Is), initial GPU ${GPU_USED_MIB} MiB ==="
echo "initial model: $INIT_MODEL"
echo "config weights: $CONFIG_WEIGHTS"
if compgen -G "$OUT/checkpoint_models/epoch*.model" >/dev/null; then
    echo "refusing an implicit restart of direct R2F training" >&2
    exit 5
fi

"$CONDA" run --no-capture-output -n "$CONDA_ENV" python \
    "$ROOT/scripts/smearing_kink/train_graphene_r2f_last_block.py" \
    --initial-model "$INIT_MODEL" \
    --train-file "$DATA/train.xyz" --valid-file "$DATA/val.xyz" \
    --test-file "$DATA/test.xyz" --output-dir "$OUT" --name "$NAME" \
    --max-epochs "$MAX_EPOCHS" --learning-rate "$LEARNING_RATE" \
    --weight-decay 1.0e-8 --ema-decay 0.99 --seed "$SEED" \
    --device cuda --eval-interval 5 \
    --checkpoint-epochs 1,5,10,20,40,80,120,160

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
echo "=== R2F last-block seed=$SEED TRAINING COMPLETE $(date -Is) ==="
