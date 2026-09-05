#!/usr/bin/env bash
# Train one compact conservative short-bond expert on frozen R2F data.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon}"
DATA="${DATA:-/data/graphene_r2f_joint_support_harmonic}"
SHORT_BASE="${SHORT_BASE:-/data/graphene_r2f_last_layer/depth3_lastblock_direct_seed83/gr_r2f_depth3_lastblock_direct_seed83.model}"
ENVIRONMENT_MODE="${ENVIRONMENT_MODE:-nonlinear}"
SEED="${SEED:-83}"
MAX_EPOCHS="${MAX_EPOCHS:-400}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
RUN_TAG="${RUN_TAG:-compact_c2_${ENVIRONMENT_MODE}}"
OUT="${OUT:-/data/graphene_r2g_short_bond/${RUN_TAG}_seed${SEED}}"
NAME="gr_r2g_${RUN_TAG}_seed${SEED}"
LOG="$OUT/run.log"

if [[ "$ENVIRONMENT_MODE" != "nonlinear" && "$ENVIRONMENT_MODE" != "constant" ]]; then
    echo "ENVIRONMENT_MODE must be nonlinear or constant" >&2
    exit 2
fi

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

if [[ -e "$OUT/TRAINING_DONE" ]]; then
    echo "R2G $RUN_TAG seed=$SEED already complete"
    exit 0
fi
for path in "$DATA/manifest.json" "$DATA/train.xyz" "$DATA/val.xyz" \
    "$DATA/test.xyz" "$DATA/supported9_train.xyz" "$SHORT_BASE"; do
    [[ -s "$path" ]]
done

GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( GPU_USED_MIB > 1000 )); then
    echo "GPU is not idle: ${GPU_USED_MIB} MiB already used" >&2
    exit 3
fi

"$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib, json, pathlib, sys
data, base, output, tag, mode, seed, epochs, lr, env, launcher, trainer, module = sys.argv[1:]
data = pathlib.Path(data); base = pathlib.Path(base); output = pathlib.Path(output)
digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
manifest = json.loads((data / "manifest.json").read_text())
assert manifest["status"] == "frozen_before_R2F_depth3_last_layer_training"
assert manifest["long_range_model_modified"] is False
assert manifest["new_DFT_labels_for_this_stage"] == 0
assert manifest["thermal_structures_used_for_gradient_updates"] == 0
payload = {
    "status": "frozen_before_training",
    "run_tag": tag,
    "environment_mode": mode,
    "architecture": {
        "energy": "C2 compact envelope times 12 radial RBFs times symmetric nonlinear endpoint environment",
        "activation_A": {"lower_off": 1.15, "lower_on": 1.20, "upper_on": 1.32, "upper_off": 1.38},
        "radial_centers_A": [1.20, 1.36],
        "n_radial": 12,
        "radial_width_A": 0.022,
        "hidden_channels": 24 if mode == "nonlinear" else 0,
    },
    "seed": int(seed), "max_epochs": int(epochs), "learning_rate": float(lr),
    "conda_environment": env,
    "short_base": {"path": str(base), "sha256": digest(base)},
    "data_manifest": {"path": str(data / "manifest.json"), "sha256": digest(data / "manifest.json")},
    "launcher": {"path": launcher, "sha256": digest(launcher)},
    "trainer": {"path": trainer, "sha256": digest(trainer)},
    "expert_module": {"path": module, "sha256": digest(module)},
    "loss": {"force_gate_scale_eV_A": 0.030, "support_energy_gate_scale_eV": 0.0194, "energy_weight": 0.25},
    "new_DFT_labels": 0,
    "thermal_gradient_configurations": 0,
    "long_range_model_modified": False,
}
output.mkdir(parents=True, exist_ok=True)
(output / "training_freeze.json").write_text(json.dumps(payload, indent=2) + "\n")
' "$DATA" "$SHORT_BASE" "$OUT" "$RUN_TAG" "$ENVIRONMENT_MODE" "$SEED" \
    "$MAX_EPOCHS" "$LEARNING_RATE" "$CONDA_ENV" \
    "$ROOT/scripts/v100/run_graphene_r2g_short_bond_expert.sh" \
    "$ROOT/scripts/smearing_kink/train_graphene_r2g_short_bond_expert.py" \
    "$ROOT/scripts/smearing_kink/graphene_short_bond_expert.py"

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

if compgen -G "$OUT/checkpoint_experts/epoch*.pt" >/dev/null; then
    echo "refusing an implicit restart of R2G training" >&2
    exit 4
fi
echo "=== R2G $RUN_TAG START $(date -Is), initial GPU ${GPU_USED_MIB} MiB ==="
"$CONDA" run --no-capture-output -n "$CONDA_ENV" python \
    "$ROOT/scripts/smearing_kink/train_graphene_r2g_short_bond_expert.py" \
    --base-model "$SHORT_BASE" \
    --train-file "$DATA/train.xyz" --valid-file "$DATA/val.xyz" \
    --test-file "$DATA/test.xyz" --manifest "$DATA/manifest.json" \
    --output-dir "$OUT" --name "$NAME" --environment-mode "$ENVIRONMENT_MODE" \
    --max-epochs "$MAX_EPOCHS" --learning-rate "$LEARNING_RATE" \
    --weight-decay 1.0e-7 --ema-decay 0.995 --energy-weight 0.25 \
    --seed "$SEED" --device cuda --eval-interval 10 \
    --checkpoint-epochs 1,5,10,20,40,80,120,160,240,320,400

MODEL="$OUT/$NAME.pt"
[[ -s "$MODEL" ]]
END_EPOCH="$(date +%s)"
"$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib, json, pathlib, sys
start, end, model, output = sys.argv[1:]
model = pathlib.Path(model); output = pathlib.Path(output)
(output / "training_runtime.json").write_text(json.dumps({
    "status": "training_complete_pending_checkpoint_gate",
    "wall_time_seconds": int(end) - int(start),
    "final_expert": str(model),
    "final_expert_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
}, indent=2) + "\n")
' "$START_EPOCH" "$END_EPOCH" "$MODEL" "$OUT"
date -Is > "$OUT/TRAINING_COMPLETED_AT"
touch "$OUT/TRAINING_DONE"
echo "=== R2G $RUN_TAG TRAINING COMPLETE $(date -Is) ==="
