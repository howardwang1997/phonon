#!/usr/bin/env bash
# Controlled same-architecture R2C short-range ablation.
set -euo pipefail

ROOT="${ROOT:-/home/howardwang/phonon}"
CONDA="${CONDA:-/home/howardwang/miniconda3/bin/conda}"
DATA="${DATA:-$ROOT/data/graphene_r2c_loss_only}"
MODE="${MODE:-gate_scaled_ef}"
SEED="${SEED:-83}"
MAX_EPOCHS="${MAX_EPOCHS:-240}"
NUM_INTERACTIONS="${NUM_INTERACTIONS:-2}"
HIDDEN_IRREPS="${HIDDEN_IRREPS:-32x0e+32x1o}"
RUN_TAG="${RUN_TAG:-$MODE}"
case "$MODE" in
    forces_only_control|gate_scaled_ef) ;;
    *) echo "MODE must be forces_only_control or gate_scaled_ef" >&2; exit 2 ;;
esac
case "$NUM_INTERACTIONS" in
    2|3) ;;
    *) echo "NUM_INTERACTIONS must be 2 or 3" >&2; exit 2 ;;
esac
case "$HIDDEN_IRREPS" in
    32x0e+32x1o|64x0e+64x1o) ;;
    *) echo "HIDDEN_IRREPS must be 32x0e+32x1o or 64x0e+64x1o" >&2; exit 2 ;;
esac

OUT="${OUT:-$ROOT/results/graphene_physics_temperature/post_p4_feasibility/R2C_short_repair/${RUN_TAG}_seed${SEED}}"
NAME="gr_r2c_${RUN_TAG}_seed${SEED}"
MODEL="$OUT/$NAME.model"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT/checkpoints" "$OUT/logs" "$OUT/results"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

if [[ -e "$OUT/TRAINING_DONE" ]]; then
    echo "R2C $MODE seed=$SEED training already complete"
    exit 0
fi
for path in "$DATA/manifest.json" "$DATA/train.xyz" "$DATA/val.xyz" \
    "$DATA/test.xyz" "$DATA/supported9_train.xyz"; do
    [[ -s "$path" ]]
done

CONFIG_WEIGHTS="$($CONDA run -n phonon python -c '
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert manifest["status"] == "frozen_before_R2C_training"
assert manifest["long_range_model_modified"] is False
assert manifest["temperature_or_degauss_is_model_input"] is False
assert manifest["new_DFT"]["n_supported"] == 9
assert manifest["new_DFT"]["n_excluded"] == 3
print(repr(manifest["config_type_weights"]))
' "$DATA/manifest.json" | tail -1)"

case "$MODE" in
    forces_only_control)
        LOSS_ARGS=(--loss forces_only --energy_weight 0.0 --forces_weight 100.0)
        ;;
    gate_scaled_ef)
        # For 72 atoms, this ratio makes 19.4 meV/config energy error and
        # 30 meV/A force error contribute comparably at the fixed R1 gates.
        LOSS_ARGS=(--loss weighted --energy_weight 12500.0 --forces_weight 1.0)
        ;;
esac

GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( GPU_USED_MIB > 1000 )); then
    echo "GPU is not idle: ${GPU_USED_MIB} MiB already used" >&2
    exit 2
fi

"$CONDA" run -n phonon python -c '
import hashlib, json, pathlib, sys
data, output, mode, run_tag, seed, epochs, num_interactions, hidden_irreps, script = sys.argv[1:]
data = pathlib.Path(data)
output = pathlib.Path(output)
script = pathlib.Path(script)
digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
manifest = json.loads((data / "manifest.json").read_text())
payload = {
    "status": "frozen_before_training",
    "mode": mode,
    "run_tag": run_tag,
    "seed": int(seed),
    "max_epochs": int(epochs),
    "architecture": {
        "num_interactions": int(num_interactions),
        "hidden_irreps": hidden_irreps,
        "r_max_A": 5.0,
        "num_radial_basis": 8,
        "num_cutoff_basis": 5,
        "correlation": 2,
    },
    "loss": (
        {"name": "forces_only", "energy_weight": 0.0, "forces_weight": 100.0}
        if mode == "forces_only_control"
        else {"name": "weighted", "energy_weight": 12500.0, "forces_weight": 1.0,
              "normalization": "R1 19.4 meV/config and 30 meV/A gates for 72 atoms"}
    ),
    "data_manifest": {"path": str(data / "manifest.json"),
                      "sha256": digest(data / "manifest.json")},
    "data_outputs": manifest["outputs"],
    "training_script": {"path": str(script), "sha256": digest(script)},
    "architecture_change": (
        int(num_interactions) != 2 or hidden_irreps != "32x0e+32x1o"
    ),
    "long_range_model_modified": False,
}
output.mkdir(parents=True, exist_ok=True)
(output / "training_freeze.json").write_text(json.dumps(payload, indent=2) + "\n")
' "$DATA" "$OUT" "$MODE" "$RUN_TAG" "$SEED" "$MAX_EPOCHS" \
    "$NUM_INTERACTIONS" "$HIDDEN_IRREPS" \
    "$ROOT/scripts/2060/run_graphene_r2c_short_ablation.sh"

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
    if (( EXIT_STATUS != 0 )); then
        touch "$OUT/FAILED"
    fi
    exit "$EXIT_STATUS"
}
trap finish EXIT

echo "=== R2C mode=$MODE run_tag=$RUN_TAG interactions=$NUM_INTERACTIONS hidden=$HIDDEN_IRREPS seed=$SEED START $(date -Is), initial GPU ${GPU_USED_MIB} MiB ==="
echo "config weights: $CONFIG_WEIGHTS"
RESTART_ARGS=()
if compgen -G "$OUT/checkpoints/${NAME}_run-*_epoch-*.pt" >/dev/null; then
    RESTART_ARGS+=(--restart_latest)
    echo "resuming from latest complete checkpoint"
fi

"$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
    --name "$NAME" --model MACE --num_interactions "$NUM_INTERACTIONS" \
    --hidden_irreps "$HIDDEN_IRREPS" --r_max 5.0 --num_radial_basis 8 \
    --num_cutoff_basis 5 --correlation 2 \
    --train_file "$DATA/train.xyz" --valid_file "$DATA/val.xyz" \
    --test_file "$DATA/test.xyz" \
    --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
    "${LOSS_ARGS[@]}" --config_type_weights "$CONFIG_WEIGHTS" \
    --batch_size 1 --valid_batch_size 1 --max_num_epochs "$MAX_EPOCHS" \
    --patience 60 --eval_interval 5 --lr 0.001 \
    --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
    --default_dtype float32 --device cuda --seed "$SEED" --save_cpu \
    --keep_checkpoints --save_all_checkpoints \
    --model_dir "$OUT" --checkpoints_dir "$OUT/checkpoints" \
    --log_dir "$OUT/logs" --results_dir "$OUT/results" \
    "${RESTART_ARGS[@]}"

[[ -s "$MODEL" ]]
END_EPOCH="$(date +%s)"
"$CONDA" run -n phonon python -c '
import hashlib, json, pathlib, sys
start, end, model, output = sys.argv[1:]
model = pathlib.Path(model)
output = pathlib.Path(output)
payload = {
    "status": "training_complete_pending_checkpoint_gate",
    "wall_time_seconds": int(end) - int(start),
    "final_model": str(model),
    "final_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
}
(output / "training_runtime.json").write_text(json.dumps(payload, indent=2) + "\n")
' "$START_EPOCH" "$END_EPOCH" "$MODEL" "$OUT"
date -Is > "$OUT/TRAINING_COMPLETED_AT"
touch "$OUT/TRAINING_DONE"
echo "=== R2C mode=$MODE run_tag=$RUN_TAG interactions=$NUM_INTERACTIONS hidden=$HIDDEN_IRREPS seed=$SEED TRAINING COMPLETE $(date -Is) ==="
