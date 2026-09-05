#!/usr/bin/env bash
# Run the fixed environment-conditioned conservative bond-order residual fit.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon}"
FIT_DATA="${FIT_DATA:-/data/graphene_r2d_support_harmonic}"
GATE_DATA="${GATE_DATA:-/data/graphene_r2d_local_adapter}"
OUT="${OUT:-/data/graphene_r2e_bond_order/environment_v1}"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

if [[ -e "$OUT/DONE" ]]; then
    echo "R2E bond-order fit already complete"
    exit 0
fi
for path in \
    "$FIT_DATA/manifest.json" "$FIT_DATA/train.xyz" \
    "$FIT_DATA/supported9_train.xyz" "$GATE_DATA/test.xyz" \
    "$GATE_DATA/val.xyz" \
    "$ROOT/data/graphene_physical_s0/operators/T300_operator.npz" \
    "$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml" \
    "$ROOT/results/graphene_physics_temperature/post_p4_feasibility/R1_order_safe_sscha/X0_cross_development/sscha_Tlat450_Tel300/result.npz"; do
    [[ -s "$path" ]]
done

GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( GPU_USED_MIB > 1000 )); then
    echo "GPU is not idle: ${GPU_USED_MIB} MiB" >&2
    exit 2
fi

"$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib, json, pathlib, sys
root, fit_data, gate_data, output, conda_env, script = sys.argv[1:]
root = pathlib.Path(root); fit_data = pathlib.Path(fit_data)
gate_data = pathlib.Path(gate_data); output = pathlib.Path(output)
script = pathlib.Path(script)
digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
paths = {
    "fit_script": root / "scripts/smearing_kink/fit_graphene_r2e_bond_order_residual.py",
    "manifest": fit_data / "manifest.json",
    "train": fit_data / "train.xyz",
    "support": fit_data / "supported9_train.xyz",
    "thermal_gate": gate_data / "test.xyz",
    "harmonic_gate": gate_data / "val.xyz",
}
payload = {
    "status": "frozen_before_R2E_fit",
    "mode": "environment_conditioned_conservative_linear_bond_order",
    "conda_environment": conda_env,
    "new_DFT_labels": 0,
    "long_range_model_modified": False,
    "inputs": {name: {"path": str(path), "sha256": digest(path)}
               for name, path in paths.items()},
    "launcher": {"path": str(script), "sha256": digest(script)},
}
output.mkdir(parents=True, exist_ok=True)
(output / "fit_freeze.json").write_text(json.dumps(payload, indent=2) + "\n")
' "$ROOT" "$FIT_DATA" "$GATE_DATA" "$OUT" "$CONDA_ENV" \
    "$ROOT/scripts/v100/run_graphene_r2e_bond_order.sh"

date -Is > "$OUT/STARTED_AT"
touch "$OUT/RUNNING"
echo "timestamp_iso,gpu_util_percent,memory_used_MiB,memory_total_MiB" > "$OUT/gpu_samples.csv"
(
    while [[ -e "$OUT/RUNNING" ]]; do
        sample="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)"
        echo "$(date -Is),$sample" >> "$OUT/gpu_samples.csv"
        sleep 10
    done
) &
MONITOR_PID=$!

finish() {
    status=$?
    trap - EXIT
    rm -f "$OUT/RUNNING"
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    echo "$status" > "$OUT/EXIT_CODE"
    if (( status != 0 )); then touch "$OUT/FAILED"; fi
    exit "$status"
}
trap finish EXIT

START_EPOCH="$(date +%s)"
"$CONDA" run --no-capture-output -n "$CONDA_ENV" python \
    "$ROOT/scripts/smearing_kink/fit_graphene_r2e_bond_order_residual.py" \
    --train-data "$FIT_DATA/train.xyz" \
    --support-data "$FIT_DATA/supported9_train.xyz" \
    --thermal-data "$GATE_DATA/test.xyz" \
    --harmonic-data "$GATE_DATA/val.xyz" \
    --manifest "$FIT_DATA/manifest.json" \
    --operator "$ROOT/data/graphene_physical_s0/operators/T300_operator.npz" \
    --background "$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml" \
    --corrected-result "$ROOT/results/graphene_physics_temperature/post_p4_feasibility/R1_order_safe_sscha/X0_cross_development/sscha_Tlat450_Tel300/result.npz" \
    --output-dir "$OUT" --device cuda --energy-balance 516.5

END_EPOCH="$(date +%s)"
"$CONDA" run -n "$CONDA_ENV" python -c '
import json, pathlib, sys
start, end, output = sys.argv[1:]
output = pathlib.Path(output)
(output / "runtime.json").write_text(json.dumps({
    "wall_time_seconds": int(end) - int(start),
    "status": "complete",
}, indent=2) + "\n")
' "$START_EPOCH" "$END_EPOCH" "$OUT"
date -Is > "$OUT/COMPLETED_AT"
touch "$OUT/DONE"
echo "R2E bond-order fit complete"
