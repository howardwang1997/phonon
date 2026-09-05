#!/usr/bin/env bash
# Train one fixed small-cutoff conservative many-body adapter on depth-3 residuals.
set -euo pipefail

ROOT="${ROOT:-/home/howardwang/phonon}"
CONDA="${CONDA:-/home/howardwang/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon}"
DATA="${DATA:-$ROOT/data/graphene_r2d_local_adapter}"
SEED="${SEED:-83}"
MAX_EPOCHS="${MAX_EPOCHS:-240}"
RUN_TAG="${RUN_TAG:-local_mb_r3_h16_c3}"
HIDDEN_IRREPS="${HIDDEN_IRREPS:-16x0e+16x1o}"
R_MAX="${R_MAX:-3.0}"
NUM_RADIAL_BASIS="${NUM_RADIAL_BASIS:-8}"
NUM_CUTOFF_BASIS="${NUM_CUTOFF_BASIS:-5}"
CORRELATION="${CORRELATION:-3}"
case "$HIDDEN_IRREPS" in
    16x0e+16x1o|32x0e+32x1o|16x0e+16x1o+16x2e) ;;
    *) echo "unsupported HIDDEN_IRREPS: $HIDDEN_IRREPS" >&2; exit 2 ;;
esac
case "$R_MAX" in
    2.0|3.0) ;;
    *) echo "unsupported R_MAX: $R_MAX" >&2; exit 2 ;;
esac
case "$NUM_RADIAL_BASIS" in
    8|16|24|32) ;;
    *) echo "unsupported NUM_RADIAL_BASIS: $NUM_RADIAL_BASIS" >&2; exit 2 ;;
esac
OUT="${OUT:-$ROOT/results/graphene_physics_temperature/post_p4_feasibility/R2D_local_adapter/${RUN_TAG}_seed${SEED}}"
NAME="gr_r2d_${RUN_TAG}_seed${SEED}"
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
    echo "R2D local adapter seed=$SEED already complete"
    exit 0
fi
for path in "$DATA/manifest.json" "$DATA/train.xyz" "$DATA/val.xyz" \
    "$DATA/test.xyz" "$DATA/supported9_train.xyz"; do
    [[ -s "$path" ]]
done

CONFIG_WEIGHTS="$($CONDA run -n "$CONDA_ENV" python -c '
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert manifest["status"] == "frozen_before_R2D_local_adapter_training"
assert manifest["long_range_model_modified"] is False
assert manifest["new_DFT_labels"] == 0
assert manifest["temperature_or_degauss_is_model_input"] is False
assert manifest["counts"]["supported9_train"] == 9
print(repr(manifest["config_type_weights"]))
' "$DATA/manifest.json" | tail -1)"

GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( GPU_USED_MIB > 1000 )); then
    echo "GPU is not idle: ${GPU_USED_MIB} MiB already used" >&2
    exit 2
fi

"$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib, json, pathlib, sys
data, output, run_tag, seed, epochs, hidden_irreps, r_max, n_radial, n_cutoff, correlation, conda_env, script = sys.argv[1:]
data = pathlib.Path(data); output = pathlib.Path(output); script = pathlib.Path(script)
digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
manifest = json.loads((data / "manifest.json").read_text())
payload = {
    "status": "frozen_before_training",
    "mode": "additive_conservative_local_many_body_residual",
    "run_tag": run_tag,
    "seed": int(seed),
    "max_epochs": int(epochs),
    "conda_environment": conda_env,
    "architecture": {
        "model": "MACE",
        "num_interactions": 2,
        "hidden_irreps": hidden_irreps,
        "r_max_A": float(r_max),
        "num_radial_basis": int(n_radial),
        "num_cutoff_basis": int(n_cutoff),
        "correlation": int(correlation),
    },
    "loss": {"name": "forces_only", "energy_weight": 0.0,
             "forces_weight": 100.0},
    "optimizer": {"lr": 0.001, "weight_decay": 1.0e-6,
                  "ema": True, "ema_decay": 0.99},
    "combination": "frozen depth-3 plus this independently trained adapter",
    "data_manifest": {"path": str(data / "manifest.json"),
                      "sha256": digest(data / "manifest.json")},
    "data_outputs": manifest["outputs"],
    "training_script": {"path": str(script), "sha256": digest(script)},
    "new_DFT_labels": 0,
    "long_range_model_modified": False,
}
output.mkdir(parents=True, exist_ok=True)
(output / "training_freeze.json").write_text(json.dumps(payload, indent=2) + "\n")
' "$DATA" "$OUT" "$RUN_TAG" "$SEED" "$MAX_EPOCHS" "$HIDDEN_IRREPS" \
    "$R_MAX" "$NUM_RADIAL_BASIS" "$NUM_CUTOFF_BASIS" "$CORRELATION" "$CONDA_ENV" \
    "$ROOT/scripts/2060/run_graphene_r2d_local_adapter.sh"

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

echo "=== R2D local adapter seed=$SEED START $(date -Is), initial GPU ${GPU_USED_MIB} MiB ==="
echo "config weights: $CONFIG_WEIGHTS"
RESTART_ARGS=()
if compgen -G "$OUT/checkpoints/${NAME}_run-*_epoch-*.pt" >/dev/null; then
    RESTART_ARGS+=(--restart_latest)
    echo "resuming from latest complete checkpoint"
fi

"$CONDA" run --no-capture-output -n "$CONDA_ENV" python -m mace.cli.run_train \
    --name "$NAME" --model MACE --num_interactions 2 \
    --hidden_irreps "$HIDDEN_IRREPS" --r_max "$R_MAX" \
    --num_radial_basis "$NUM_RADIAL_BASIS" \
    --num_cutoff_basis "$NUM_CUTOFF_BASIS" --correlation "$CORRELATION" \
    --train_file "$DATA/train.xyz" --valid_file "$DATA/val.xyz" \
    --test_file "$DATA/test.xyz" \
    --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
    --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
    --config_type_weights "$CONFIG_WEIGHTS" \
    --batch_size 1 --valid_batch_size 1 --max_num_epochs "$MAX_EPOCHS" \
    --patience 60 --eval_interval 5 --lr 0.001 \
    --weight_decay 1.0e-6 --ema --ema_decay 0.99 \
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
    "status": "training_complete_pending_additive_gate",
    "wall_time_seconds": int(end) - int(start),
    "final_model": str(model),
    "final_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
}, indent=2) + "\n")
' "$START_EPOCH" "$END_EPOCH" "$MODEL" "$OUT"
date -Is > "$OUT/TRAINING_COMPLETED_AT"
touch "$OUT/TRAINING_DONE"
echo "=== R2D local adapter seed=$SEED TRAINING COMPLETE $(date -Is) ==="
