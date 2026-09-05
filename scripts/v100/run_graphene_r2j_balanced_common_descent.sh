#!/usr/bin/env bash
# Gate-balanced common-descent fine-tune of one or two terminal depth-3 blocks.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon}"
DATA="${DATA:-/data/graphene_r2f_joint_support_harmonic}"
EVAL_DATA="${EVAL_DATA:-/data/graphene_r2c_eval}"
INIT_MODEL="${INIT_MODEL:-/data/graphene_r2c_short_repair/depth3_forces_only_seed83/gr_r2c_depth3_forces_only_seed83.model}"
BASE_MODEL="${BASE_MODEL:-$ROOT/results/gr_backbone_v11/ft_graphene.model}"
BACKGROUND="${BACKGROUND:-$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml}"
CORRECTED_RESULT="${CORRECTED_RESULT:-$EVAL_DATA/corrected_result.npz}"
SCOPE="${SCOPE:-last_block}"
SEED="${SEED:-83}"
MAX_EPOCHS="${MAX_EPOCHS:-240}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-1,5,10,20,40,80,120,160,200,240}"
RUN_TAG="${RUN_TAG:-balanced_${SCOPE}}"
OUT="${OUT:-/data/graphene_r2j_common_descent/${RUN_TAG}_seed${SEED}}"
NAME="${NAME:-gr_r2j_${RUN_TAG}_seed${SEED}}"
MODEL="$OUT/$NAME.model"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

if [[ -e "$OUT/TRAINING_DONE" ]]; then
    echo "R2J $RUN_TAG seed=$SEED already complete"
    exit 0
fi
for path in "$DATA/manifest.json" "$DATA/train.xyz" "$DATA/val.xyz" \
    "$DATA/test.xyz" "$DATA/supported9_train.xyz" "$INIT_MODEL" "$BASE_MODEL" \
    "$EVAL_DATA/operators/T300_operator.npz" "$BACKGROUND" "$CORRECTED_RESULT"; do
    [[ -s "$path" ]]
done
case "$SCOPE" in
    last_block|last_two_blocks) ;;
    *) echo "invalid R2J scope: $SCOPE" >&2; exit 2 ;;
esac

GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( GPU_USED_MIB > 1000 )); then
    echo "GPU is not idle: ${GPU_USED_MIB} MiB already used" >&2
    exit 3
fi
if compgen -G "$OUT/checkpoint_models/epoch*.model" >/dev/null; then
    echo "refusing an implicit restart of R2J training" >&2
    exit 4
fi

"$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib, json, pathlib, sys
data, initial, output, scope, tag, name, seed, epochs, lr, checkpoints, script, trainer = sys.argv[1:]
data = pathlib.Path(data); initial = pathlib.Path(initial); output = pathlib.Path(output)
digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
manifest = json.loads((data / "manifest.json").read_text())
assert manifest["long_range_model_modified"] is False
assert manifest["new_DFT_labels_for_this_stage"] == 0
assert manifest["thermal_structures_used_for_gradient_updates"] == 0
assert manifest["counts"] == {"train": 81, "val": 34, "test": 26, "supported9_train": 9}
payload = {
    "status": "frozen_before_R2J_training",
    "method": "gate_normalized_balanced_common_descent",
    "scope": scope,
    "run_tag": tag,
    "name": name,
    "seed": int(seed),
    "max_epochs": int(epochs),
    "learning_rate": float(lr),
    "checkpoint_epochs": [int(v) for v in checkpoints.split(",") if v],
    "initial_model": {"path": str(initial), "sha256": digest(initial)},
    "data_manifest": {"path": str(data / "manifest.json"), "sha256": digest(data / "manifest.json")},
    "training_script": {"path": script, "sha256": digest(script)},
    "trainer": {"path": trainer, "sha256": digest(trainer)},
    "new_DFT_labels": 0,
    "long_range_model_modified": False,
    "thermal_structures_used_for_gradient_updates": 0,
}
output.mkdir(parents=True, exist_ok=True)
(output / "training_freeze.json").write_text(json.dumps(payload, indent=2) + "\n")
' "$DATA" "$INIT_MODEL" "$OUT" "$SCOPE" "$RUN_TAG" "$NAME" "$SEED" \
    "$MAX_EPOCHS" "$LEARNING_RATE" "$CHECKPOINT_EPOCHS" \
    "$ROOT/scripts/v100/run_graphene_r2j_balanced_common_descent.sh" \
    "$ROOT/scripts/smearing_kink/train_graphene_r2j_balanced_common_descent.py"

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

echo "=== R2J $RUN_TAG START $(date -Is), scope=$SCOPE, initial GPU ${GPU_USED_MIB} MiB ==="
"$CONDA" run --no-capture-output -n "$CONDA_ENV" python \
    "$ROOT/scripts/smearing_kink/train_graphene_r2j_balanced_common_descent.py" \
    --initial-model "$INIT_MODEL" \
    --train-file "$DATA/train.xyz" --valid-file "$DATA/val.xyz" \
    --test-file "$DATA/test.xyz" \
    --operator "$EVAL_DATA/operators/T300_operator.npz" \
    --background "$BACKGROUND" --corrected-result "$CORRECTED_RESULT" \
    --output-dir "$OUT" --name "$NAME" --scope "$SCOPE" \
    --max-epochs "$MAX_EPOCHS" --learning-rate "$LEARNING_RATE" \
    --weight-decay 1.0e-8 --ema-decay 0.99 --top-k 16 --seed "$SEED" \
    --device cuda --eval-interval 10 --checkpoint-epochs "$CHECKPOINT_EPOCHS"

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
echo "=== R2J $RUN_TAG TRAINING COMPLETE $(date -Is) ==="
