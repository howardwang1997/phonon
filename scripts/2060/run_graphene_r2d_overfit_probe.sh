#!/usr/bin/env bash
# Cheap diagnostic: can the fixed R2D adapter fit one/all supported structures?
set -euo pipefail

ROOT="${ROOT:-/home/howardwang/phonon}"
CONDA="${CONDA:-/home/howardwang/miniconda3/bin/conda}"
MODE="${MODE:?MODE must be sscha99 or support9}"
SEED="${SEED:-83}"
case "$MODE" in
    sscha99) MAX_EPOCHS="${MAX_EPOCHS:-120}" ;;
    support9) MAX_EPOCHS="${MAX_EPOCHS:-180}" ;;
    *) echo "MODE must be sscha99 or support9" >&2; exit 2 ;;
esac

DATA_FILE="${DATA_FILE:-$ROOT/data/graphene_r2d_overfit_probe/${MODE}.xyz}"
OUT="${OUT:-$ROOT/results/graphene_physics_temperature/post_p4_feasibility/R2D_local_adapter/overfit_${MODE}_seed${SEED}}"
NAME="gr_r2d_overfit_${MODE}_seed${SEED}"
MODEL="$OUT/$NAME.model"

cd "$ROOT"
mkdir -p "$OUT/checkpoints" "$OUT/logs" "$OUT/results"
exec > >(tee -a "$OUT/run.log") 2>&1
export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

[[ -s "$DATA_FILE" ]]
if [[ -e "$OUT/DONE" ]]; then
    echo "$MODE overfit probe already complete"
    exit 0
fi
GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( GPU_USED_MIB > 1000 )); then
    echo "GPU is not idle: ${GPU_USED_MIB} MiB" >&2
    exit 3
fi

"$CONDA" run -n phonon python -c '
import hashlib, json, pathlib, sys
data, output, mode, seed, epochs, script = sys.argv[1:]
data = pathlib.Path(data); output = pathlib.Path(output); script = pathlib.Path(script)
digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
payload = {
    "status": "frozen_before_overfit_probe",
    "scope": "expressibility/optimization diagnostic only; not a selectable model",
    "mode": mode, "seed": int(seed), "max_epochs": int(epochs),
    "architecture": {"num_interactions": 2, "hidden_irreps": "16x0e+16x1o",
                     "r_max_A": 3.0, "num_radial_basis": 8,
                     "num_cutoff_basis": 5, "correlation": 3},
    "loss": {"name": "forces_only", "forces_weight": 100.0},
    "data": {"path": str(data), "sha256": digest(data)},
    "script": {"path": str(script), "sha256": digest(script)},
    "new_DFT_labels": 0, "long_range_model_modified": False,
}
output.mkdir(parents=True, exist_ok=True)
(output / "probe_freeze.json").write_text(json.dumps(payload, indent=2) + "\n")
' "$DATA_FILE" "$OUT" "$MODE" "$SEED" "$MAX_EPOCHS" \
    "$ROOT/scripts/2060/run_graphene_r2d_overfit_probe.sh"

date -Is > "$OUT/STARTED_AT"
touch "$OUT/RUNNING"
finish() {
    status=$?
    trap - EXIT
    rm -f "$OUT/RUNNING"
    echo "$status" > "$OUT/EXIT_CODE"
    if (( status != 0 )); then touch "$OUT/FAILED"; fi
    exit "$status"
}
trap finish EXIT

"$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
    --name "$NAME" --model MACE --num_interactions 2 \
    --hidden_irreps '16x0e+16x1o' --r_max 3.0 --num_radial_basis 8 \
    --num_cutoff_basis 5 --correlation 3 \
    --train_file "$DATA_FILE" --valid_file "$DATA_FILE" --test_file "$DATA_FILE" \
    --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
    --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
    --config_type_weights "{'r2c_fixed_smearing_supported':1.0,'Default':1.0}" \
    --batch_size 1 --valid_batch_size 1 --max_num_epochs "$MAX_EPOCHS" \
    --patience 300 --eval_interval 10 --lr 0.001 --weight_decay 1.0e-6 \
    --ema --ema_decay 0.99 --default_dtype float32 --device cuda --seed "$SEED" \
    --save_cpu --keep_checkpoints --save_all_checkpoints \
    --model_dir "$OUT" --checkpoints_dir "$OUT/checkpoints" \
    --log_dir "$OUT/logs" --results_dir "$OUT/results"

[[ -s "$MODEL" ]]
date -Is > "$OUT/COMPLETED_AT"
touch "$OUT/DONE"
echo "R2D $MODE overfit probe complete"
