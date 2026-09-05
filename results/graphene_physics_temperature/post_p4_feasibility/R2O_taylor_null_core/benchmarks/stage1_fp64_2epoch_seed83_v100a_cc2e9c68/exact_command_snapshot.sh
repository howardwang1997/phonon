#!/usr/bin/env bash
set -euo pipefail

# Isolated throughput/CLI benchmark only.  This is intentionally outside every
# formal R2O output path and never reads validation, seed2, or support data.
CONDA=/root/miniconda3/bin/conda
CONDA_ENV=phonon-mlip
BENCH_ROOT=/data/graphene_r2o_stage1_benchmark
INPUT="$BENCH_ROOT/input_cc2e9c68"
TRAIN="$INPUT/train_thermal.xyz"
RECEIPT="$INPUT/input_receipt.json"
OUT="$BENCH_ROOT/stage1_fp64_2epoch_seed83_v100a_cc2e9c68"
NAME=r2o_stage1_fp64_2epoch_cli_benchmark_seed83
EXPECTED_TRAIN_SHA=cc2e9c68d418fba996e8ac89c8156c8adf00768406f6171aaa57f0a9bd2720c1

if [[ -e "$OUT" ]]; then
    echo "refusing to overwrite existing isolated benchmark: $OUT" >&2
    exit 2
fi
[[ -s "$TRAIN" && -s "$RECEIPT" ]]
[[ "$(sha256sum "$TRAIN" | awk '{print $1}')" == "$EXPECTED_TRAIN_SHA" ]]

mkdir -p "$OUT/checkpoints" "$OUT/logs" "$OUT/results"
cp "$0" "$OUT/exact_command_snapshot.sh"
cp "$RECEIPT" "$OUT/input_receipt.json"
printf '%s\n' \
  'BENCHMARK ONLY: not a formal R2O model, checkpoint gate, or deployment artifact.' \
  > "$OUT/BENCHMARK_ONLY_NOT_FORMAL.txt"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR=/tmp/matplotlib-r2o-stage1-benchmark
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

{
  hostname
  date -Is
  "$CONDA" run --no-capture-output -n "$CONDA_ENV" python -c \
    'import json,platform,torch,importlib.metadata as m; print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"mace_torch":m.version("mace-torch"),"ase":m.version("ase"),"scipy":m.version("scipy"),"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0)}))'
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader
} > "$OUT/environment.txt" 2>&1

echo 'timestamp_iso,gpu_util_percent,memory_used_MiB,memory_total_MiB,power_draw_W' > "$OUT/gpu_samples.csv"
(
  while :; do
    sample="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits)"
    echo "$(date -Is),$sample" >> "$OUT/gpu_samples.csv"
    sleep 0.2
  done
) &
MONITOR_PID=$!

START_EPOCH="$(date +%s)"
set +e
"$CONDA" run --no-capture-output -n "$CONDA_ENV" python -m mace.cli.run_train \
  --name "$NAME" --model MACE \
  --num_interactions 2 --hidden_irreps '16x0e+16x1o+16x2e' \
  --r_max 3.2 --num_radial_basis 24 --num_cutoff_basis 5 \
  --max_ell 2 --correlation 3 \
  --train_file "$TRAIN" --valid_file "$TRAIN" \
  --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
  --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
  --config_type_weights "{'r2o_exact_e50_seed0_train': 1.0, 'r2o_auxiliary_T300_train': 0.2777777777777778, 'r2o_auxiliary_T600_train': 0.2777777777777778}" \
  --batch_size 1 --valid_batch_size 1 --max_num_epochs 2 \
  --patience 10 --eval_interval 1 --lr 0.001 --weight_decay 1.0e-6 \
  --scheduler ExponentialLR --lr_scheduler_gamma 1.0 \
  --default_dtype float64 --device cuda --seed 83 \
  --save_cpu --keep_checkpoints --save_all_checkpoints \
  --model_dir "$OUT" --checkpoints_dir "$OUT/checkpoints" \
  --log_dir "$OUT/logs" --results_dir "$OUT/results" \
  2>&1 | tee "$OUT/run.log"
MACE_STATUS=${PIPESTATUS[0]}
set -e
END_EPOCH="$(date +%s)"
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
printf '%s\n' "$MACE_STATUS" > "$OUT/command_exit_code.txt"
if (( MACE_STATUS != 0 )); then
  exit "$MACE_STATUS"
fi

MODEL="$OUT/$NAME.model"
CHECKPOINT="$OUT/checkpoints/${NAME}_run-83_epoch-1.pt"
[[ -s "$MODEL" && -s "$CHECKPOINT" ]]

"$CONDA" run --no-capture-output -n "$CONDA_ENV" python -c '
import collections, datetime, hashlib, json, pathlib, re, sys
import numpy as np
import torch
from ase.io import read

out = pathlib.Path(sys.argv[1])
train = pathlib.Path(sys.argv[2])
model_path = pathlib.Path(sys.argv[3])
checkpoint_path = pathlib.Path(sys.argv[4])
start_s, end_s = map(int, sys.argv[5:7])
digest = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
model = torch.load(model_path, map_location="cpu", weights_only=False)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state = model.state_dict()
checkpoint_state = checkpoint["model"]
same_keys = set(state) == set(checkpoint_state)
state_exact = same_keys and all(torch.equal(state[name].cpu(), checkpoint_state[name].cpu()) for name in state)
steps = [float(record["step"]) for record in checkpoint["optimizer"]["state"].values() if "step" in record]
frames = read(train, index=":")
counts = collections.Counter(str(frame.info["config_type"]) for frame in frames)
layout = []
width = 0
for interaction, product in enumerate(model.products):
    for multiplicity, irrep in product.linear.irreps_out:
        layout.append([interaction, int(multiplicity), int(irrep.l), int(irrep.p)])
        width += int(multiplicity) * int(irrep.dim)
samples = []
for line in (out / "gpu_samples.csv").read_text().splitlines()[1:]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) == 5:
        samples.append({"util": float(parts[1]), "used": float(parts[2]), "total": float(parts[3]), "power": float(parts[4])})
timestamps = []
pattern = re.compile(r"^(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d+) INFO: Epoch ([01]):")
for line in (out / "run.log").read_text(errors="replace").splitlines():
    match = pattern.match(line)
    if match:
        timestamps.append((int(match.group(2)), datetime.datetime.fromisoformat(match.group(1))))
epoch_interval = None
by_epoch = {epoch: stamp for epoch, stamp in timestamps}
if 0 in by_epoch and 1 in by_epoch:
    epoch_interval = (by_epoch[1] - by_epoch[0]).total_seconds()
payload = {
  "format": "r2o_stage1_fp64_cli_2epoch_benchmark_v1",
  "scope": "isolated throughput/CLI semantics only; not a formal R2O artifact or model gate",
  "command_exit_code": 0,
  "train_sha256": digest(train),
  "train_and_valid_are_the_same_allowed_thermal_file": True,
  "configuration_counts": dict(sorted(counts.items())),
  "training_configurations": len(frames),
  "zero_based_epochs": [0, 1],
  "expected_optimizer_steps": 184,
  "optimizer_step_min": min(steps),
  "optimizer_step_max": max(steps),
  "optimizer_step_count_matches": min(steps) == max(steps) == 184.0,
  "total_command_wall_seconds": end_s - start_s,
  "epoch0_to_epoch1_log_interval_seconds": epoch_interval,
  "gpu_sample_count": len(samples),
  "gpu_peak_memory_used_MiB_nvidia_smi": max(item["used"] for item in samples),
  "gpu_peak_utilization_percent": max(item["util"] for item in samples),
  "gpu_peak_power_draw_W": max(item["power"] for item in samples),
  "architecture": {
    "class": model.__class__.__name__,
    "dtype": str(next(model.parameters()).dtype),
    "r_max_A": float(model.r_max),
    "num_interactions": int(model.num_interactions),
    "hidden_layout": layout,
    "raw_node_width": width,
    "num_radial_basis": int(model.radial_embedding.bessel_fn.bessel_weights.numel()),
    "cutoff_p": int(model.radial_embedding.cutoff_fn.p),
    "pair_repulsion_present": hasattr(model, "pair_repulsion"),
    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
  },
  "optimizer_scheduler": {
    "optimizer": "AdamW",
    "lr": checkpoint["optimizer"]["param_groups"][0]["lr"],
    "weight_decay": checkpoint["optimizer"]["param_groups"][0]["weight_decay"],
    "scheduler_gamma": checkpoint["lr_scheduler"]["gamma"],
    "scheduler_last_epoch": checkpoint["lr_scheduler"]["last_epoch"],
    "scheduler_step_count": checkpoint["lr_scheduler"]["_step_count"],
    "ema_enabled": False,
    "swa_enabled": False,
  },
  "final_model_sha256": digest(model_path),
  "epoch1_checkpoint_sha256": digest(checkpoint_path),
  "final_state_equals_epoch1_raw_checkpoint_state_exactly": state_exact,
  "checkpoint_and_final_state_keys_identical": same_keys,
  "no_seed1_seed2_support_or_prior_model_read": True,
  "run_log_contains_traceback": "Traceback" in (out / "run.log").read_text(errors="replace"),
}
if not payload["optimizer_step_count_matches"] or not state_exact:
    raise RuntimeError("stage1 benchmark optimizer-step or raw-state audit failed")
if payload["architecture"] != {
    "class": "ScaleShiftMACE", "dtype": "torch.float64", "r_max_A": 3.2,
    "num_interactions": 2,
    "hidden_layout": [[0,16,0,1],[0,16,1,-1],[0,16,2,1],[1,16,0,1]],
    "raw_node_width": 160, "num_radial_basis": 24, "cutoff_p": 5,
    "pair_repulsion_present": False, "parameter_count": 42096,
}:
    raise RuntimeError("stage1 architecture audit failed: %r" % payload["architecture"])
(out / "benchmark_result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\\n")
print(json.dumps(payload, indent=2, allow_nan=False))
' "$OUT" "$TRAIN" "$MODEL" "$CHECKPOINT" "$START_EPOCH" "$END_EPOCH"

sha256sum "$OUT/exact_command_snapshot.sh" "$OUT/input_receipt.json" \
  "$MODEL" "$CHECKPOINT" "$OUT/benchmark_result.json" \
  > "$OUT/artifact_sha256.txt"
cat "$OUT/benchmark_result.json"
