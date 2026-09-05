#!/usr/bin/env bash
# Train the one frozen R2M support-free replacement-core architecture.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon}"
DATA="${DATA:-$ROOT/data/graphene_r2m_support_free_core}"
OUT="${OUT:-/data/graphene_r2m_core/support_free_r2_h16_l2_n24_seed83}"
NAME="gr_r2m_support_free_r2_h16_l2_n24_seed83"
MODEL="$OUT/$NAME.model"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT/checkpoints" "$OUT/logs" "$OUT/results" "$OUT/code_snapshots"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r2m-core}"

if [[ -e "$OUT/DONE" ]]; then
    echo "R2M support-free core already complete"
    exit 0
fi

RECOVER_AFTER_TRAINING=0
if [[ -e "$OUT/TRAINING_DONE" ]]; then
    for path in "$MODEL" "$OUT/training_runtime.json" "$OUT/training_freeze.json" \
        "$OUT/code_snapshots/evaluate_graphene_r2m_core_checkpoints.py"; do
        [[ -s "$path" ]]
    done
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,json,pathlib,sys
model,runtime_path,freeze_path,evaluator_path,data_manifest = map(pathlib.Path,sys.argv[1:])
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
runtime=json.loads(runtime_path.read_text())
freeze=json.loads(freeze_path.read_text())
assert runtime["status"] == "training_complete_pending_fixed_checkpoint_gate"
assert pathlib.Path(runtime["final_model"]).resolve() == model.resolve()
assert runtime["final_model_sha256"] == digest(model)
assert freeze["status"] == "frozen_before_R2M_core_training"
assert freeze["evaluator"]["sha256"] == digest(evaluator_path)
assert freeze["data_manifest"]["sha256"] == digest(data_manifest)
' "$MODEL" "$OUT/training_runtime.json" "$OUT/training_freeze.json" \
        "$OUT/code_snapshots/evaluate_graphene_r2m_core_checkpoints.py" \
        "$DATA/manifest.json"
    RECOVER_AFTER_TRAINING=1
elif compgen -G "$OUT/checkpoints/*_epoch-*.pt" >/dev/null; then
    echo "refusing implicit restart: checkpoint files already exist" >&2
    exit 2
fi
for path in "$DATA/manifest.json" "$DATA/train.xyz" "$DATA/valid.xyz" \
    "$DATA/reserved_e50_seed2.xyz"; do
    [[ -s "$path" ]]
done
if (( RECOVER_AFTER_TRAINING == 0 )); then
    [[ -s "$ROOT/scripts/smearing_kink/evaluate_graphene_r2m_core_checkpoints.py" ]]
fi

CONFIG_WEIGHTS="$($CONDA run -n "$CONDA_ENV" python -c '
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]); manifest = json.loads((root / "manifest.json").read_text())
assert manifest["status"] == "frozen_before_R2M_support_free_core_training"
assert manifest["model_role"] == "replacement_short_delta_core"
assert manifest["depth3_or_current_S0_in_deployed_model"] is False
assert manifest["energy_training_enabled"] is False
assert manifest["counts"] == {
 "train":164,"train_exact_e50":20,"train_harmonic":72,
 "train_auxiliary_T300":36,"train_auxiliary_T600":36,"valid":45,
 "valid_exact_e50_seed1":20,"valid_harmonic":25,
 "reserved_e50_seed2":20,"support_used":0}
assert manifest["architecture_freeze"]["new_core_no_periodic_wrap_by_bound"] is True
assert manifest["architecture_freeze"]["interaction_diameter_bound_A"] == 8.0
assert manifest["leakage_control"]["seed1_in_gradients_or_scales"] is False
assert manifest["leakage_control"]["seed2_in_gradients_scales_or_checkpoint_selection"] is False
for name in ("train.xyz", "valid.xyz", "reserved_e50_seed2.xyz", "prepare_script_snapshot.py"):
 p=root/name; expected=manifest["outputs"][name]["sha256"]
 assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, name
print(repr(manifest["loss_weighting"]["normalized_config_type_weights"]))
' "$DATA" | tail -1)"

GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( GPU_USED_MIB > 1000 )); then
    echo "GPU is not idle: ${GPU_USED_MIB} MiB already used" >&2
    exit 3
fi

if (( RECOVER_AFTER_TRAINING == 0 )); then
    cp "$ROOT/scripts/v100/run_graphene_r2m_support_free_core.sh" \
        "$OUT/code_snapshots/run_graphene_r2m_support_free_core.sh"
    cp "$DATA/prepare_script_snapshot.py" "$OUT/code_snapshots/prepare_script_snapshot.py"
    cp "$ROOT/scripts/smearing_kink/evaluate_graphene_r2m_core_checkpoints.py" \
        "$OUT/code_snapshots/evaluate_graphene_r2m_core_checkpoints.py"

    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib, inspect, json, pathlib, sys
import mace
from mace.modules.loss import mean_squared_error_forces
data, output, environment, launcher = map(pathlib.Path, sys.argv[1:5])
manifest = json.loads((data / "manifest.json").read_text())
digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
loss_source = inspect.getsource(mean_squared_error_forces)
for required in (
 "torch.repeat_interleave", "ref.weight", "ref.forces_weight",
 "torch.square", "reduce_loss(raw_loss",
):
 assert required in loss_source, required
payload = {
 "status":"frozen_before_R2M_core_training",
 "model_role":"replacement_short_delta_core",
 "combination":"frozen_v11_foundation + new_R2M_core + frozen_q6",
 "depth3_or_current_S0_initialized_or_deployed":False,
 "random_initialization":True,
 "architecture":{"model":"MACE","num_interactions":2,"r_max_A":2.0,
  "hidden_irreps":"16x0e+16x1o+16x2e","max_ell":2,
  "num_radial_basis":24,"num_cutoff_basis":5,"correlation":3,
  "interaction_diameter_bound_A":8.0},
 "loss":{"name":"forces_only","energy_weight":0.0,"forces_weight":100.0,
  "config_type_weights":manifest["loss_weighting"]["normalized_config_type_weights"],
  "batch1_semantics":"each complete configuration is one optimizer step; MACE repeats the configuration and force weights over its atoms, squares component errors, then reduces",
  "mace_version":getattr(mace,"__version__","unknown"),
  "mean_squared_error_forces_source_sha256":hashlib.sha256(loss_source.encode()).hexdigest()},
 "optimizer":{"lr":0.001,"weight_decay":1e-6,"ema":True,"ema_decay":0.99,
  "scheduler":"ExponentialLR","lr_scheduler_gamma":1.0,
  "validation_data_controls_lr":False},
 "training":{"seed":83,"batch_size":1,"max_epochs":240,"patience":300,
  "fixed_evaluation_epochs":[40,80,120,160,200,235],
  "final_is_also_evaluated":True,
  "validation_loss_cannot_stop_training":True},
 "leakage":{"support_in_gradients":False,"E50_seed1_in_gradients":False,
  "E50_seed2_read_by_training":False,"atom_or_bond_random_split":False},
 "data_manifest":{"path":str(data / "manifest.json"),"sha256":digest(data / "manifest.json")},
 "launcher":{"path":str(launcher),"sha256":digest(launcher)},
 "evaluator":{"path":str(output / "code_snapshots/evaluate_graphene_r2m_core_checkpoints.py"),
  "sha256":digest(output / "code_snapshots/evaluate_graphene_r2m_core_checkpoints.py")},
 "conda_environment":str(environment),
}
(output / "training_freeze.json").write_text(json.dumps(payload,indent=2)+"\n")
' "$DATA" "$OUT" "$CONDA_ENV" \
        "$OUT/code_snapshots/run_graphene_r2m_support_free_core.sh"
fi

if (( RECOVER_AFTER_TRAINING == 0 )); then
    date -Is > "$OUT/STARTED_AT"
else
    date -Is > "$OUT/EVALUATION_RECOVERY_STARTED_AT"
fi
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

if (( RECOVER_AFTER_TRAINING == 0 )); then
    START_EPOCH="$(date +%s)"
    echo "=== R2M support-free replacement core TRAIN START $(date -Is) ==="
    echo "config weights: $CONFIG_WEIGHTS"
    "$CONDA" run --no-capture-output -n "$CONDA_ENV" python -m mace.cli.run_train \
    --name "$NAME" --model MACE --num_interactions 2 \
    --hidden_irreps '16x0e+16x1o+16x2e' --r_max 2.0 \
    --num_radial_basis 24 --num_cutoff_basis 5 --max_ell 2 --correlation 3 \
    --train_file "$DATA/train.xyz" --valid_file "$DATA/valid.xyz" \
    --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
    --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
    --config_type_weights "$CONFIG_WEIGHTS" \
    --batch_size 1 --valid_batch_size 1 --max_num_epochs 240 \
    --patience 300 --eval_interval 5 --lr 0.001 --weight_decay 1.0e-6 \
    --scheduler ExponentialLR --lr_scheduler_gamma 1.0 \
    --ema --ema_decay 0.99 --default_dtype float32 --device cuda --seed 83 \
    --save_cpu --keep_checkpoints --save_all_checkpoints \
    --model_dir "$OUT" --checkpoints_dir "$OUT/checkpoints" \
        --log_dir "$OUT/logs" --results_dir "$OUT/results"

    [[ -s "$MODEL" ]]
    END_EPOCH="$(date +%s)"
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib, json, pathlib, sys
start,end,model,output=sys.argv[1:]; model=pathlib.Path(model); output=pathlib.Path(output)
(output/"training_runtime.json").write_text(json.dumps({
 "status":"training_complete_pending_fixed_checkpoint_gate",
 "wall_time_seconds":int(end)-int(start),"final_model":str(model),
 "final_model_sha256":hashlib.sha256(model.read_bytes()).hexdigest()},indent=2)+"\n")
' "$START_EPOCH" "$END_EPOCH" "$MODEL" "$OUT"
    date -Is > "$OUT/TRAINING_COMPLETED_AT"
    touch "$OUT/TRAINING_DONE"
else
    echo "=== R2M training already complete; recovering fixed checkpoint gate $(date -Is) ==="
fi
"$CONDA" run --no-capture-output -n "$CONDA_ENV" python \
    "$OUT/code_snapshots/evaluate_graphene_r2m_core_checkpoints.py" \
    --data "$DATA" --training-dir "$OUT" --template-model "$MODEL" \
    --device cuda --output "$OUT/core_checkpoint_gate.json" \
    --selected-model "$OUT/selected_core.model"
GATE_STATUS="$($CONDA run -n "$CONDA_ENV" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$OUT/core_checkpoint_gate.json" | tail -1)"
if [[ "$GATE_STATUS" == "R2M_core_checkpoint_gate_passed" ]]; then
    rm -f "$OUT/CORE_GATE_FAILED"
    touch "$OUT/CORE_GATE_PASSED"
else
    rm -f "$OUT/CORE_GATE_PASSED"
    touch "$OUT/CORE_GATE_FAILED"
fi
date -Is > "$OUT/COMPLETED_AT"
rm -f "$OUT/FAILED"
touch "$OUT/DONE"
echo "=== R2M support-free replacement core COMPLETE gate=$GATE_STATUS $(date -Is) ==="
