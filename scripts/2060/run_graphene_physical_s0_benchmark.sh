#!/usr/bin/env bash
# Ten-epoch runtime/learning benchmark for the three-temperature S0 delta MACE.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DATA="$ROOT/data/graphene_physical_s0"
MANIFEST="$DATA/manifest.json"
BASE="$ROOT/results/gr_backbone_v11/ft_graphene.model"
OUT="$ROOT/results/graphene_physics_temperature/post_p4_feasibility/S0_unified_short/benchmark_10ep_2060"
NAME="gr_physical_s0_delta32_benchmark10"
MODEL="$OUT/$NAME.model"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT/checkpoints" "$OUT/logs" "$OUT/results"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$OUT/DONE" ]]; then
    echo "S0 ten-epoch benchmark already complete"
    exit 0
fi
for path in "$MANIFEST" "$BASE" "$DATA/train.xyz" "$DATA/val.xyz" \
    "$DATA/test.xyz" "$DATA/T300/test.xyz" "$DATA/T450/test.xyz" \
    "$DATA/T600/test.xyz" "$DATA/replay/val.xyz"; do
    [[ -s "$path" ]]
done

"$CONDA" run -n phonon python -c '
import hashlib, json, pathlib, sys
manifest_path, base_path = map(pathlib.Path, sys.argv[1:])
m = json.loads(manifest_path.read_text())
h = hashlib.sha256(base_path.read_bytes()).hexdigest()
assert m["status"] == "passed"
assert all(m["aggregate_gates"]["checks"].values())
assert m["development_temperatures_K"] == [300, 450, 600]
assert m["locked_validation_temperatures_K_not_accessed"] == [375, 525]
assert m["temperature_or_degauss_is_model_input"] is False
assert h == m["base_model_sha256"]
' "$MANIFEST" "$BASE"

CONFIG_WEIGHTS="$($CONDA run -n phonon python -c '
import json, sys
m = json.load(open(sys.argv[1]))
print(repr(m["recommended_config_weights"]["config_type_weights"]))
' "$MANIFEST" | tail -1)"

S0_GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( S0_GPU_USED_MIB > 1000 )); then
    echo "GPU is no longer idle: ${S0_GPU_USED_MIB} MiB already used"
    exit 2
fi

S0_START_EPOCH="$(date +%s)"
date -Is > "$OUT/STARTED_AT"
touch "$OUT/RUNNING"
echo "=== S0 ten-epoch benchmark START $(date -Is), initial GPU ${S0_GPU_USED_MIB} MiB ==="
echo "config weights: $CONFIG_WEIGHTS"

"$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
    --name "$NAME" --model MACE --num_interactions 2 \
    --hidden_irreps '32x0e+32x1o' --r_max 5.0 --num_radial_basis 8 \
    --num_cutoff_basis 5 --correlation 2 \
    --train_file "$DATA/train.xyz" --valid_file "$DATA/val.xyz" \
    --test_file "$DATA/test.xyz" \
    --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
    --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
    --config_type_weights "$CONFIG_WEIGHTS" \
    --batch_size 1 --valid_batch_size 1 --max_num_epochs 10 \
    --patience 20 --eval_interval 5 --lr 0.001 \
    --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
    --default_dtype float32 --device cuda --seed 83 --save_cpu \
    --keep_checkpoints --save_all_checkpoints \
    --model_dir "$OUT" --checkpoints_dir "$OUT/checkpoints" \
    --log_dir "$OUT/logs" --results_dir "$OUT/results"

[[ -s "$MODEL" ]]
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_fd_joint_checkpoints.py \
    --base-model "$BASE" --template-model "$MODEL" \
    --checkpoint-dir "$OUT/checkpoints" \
    --thermal "thermal300=$DATA/T300/test.xyz" \
    --thermal "thermal450=$DATA/T450/test.xyz" \
    --thermal "thermal600=$DATA/T600/test.xyz" \
    --harmonic "$DATA/replay/val.xyz" --device cuda \
    --force-rmse-threshold 50 --force-max-threshold 250 \
    --harmonic-rmse-ratio 2 \
    --scope "S0 ten-epoch runtime and learning benchmark; not final checkpoint selection" \
    --output "$OUT/checkpoint_sweep.json" \
    --selected-model "$OUT/best_10ep_checkpoint.model"

S0_END_EPOCH="$(date +%s)"
"$CONDA" run -n phonon python -c '
import json, pathlib, sys
start, end = map(int, sys.argv[1:3])
path = pathlib.Path(sys.argv[3])
path.write_text(json.dumps({
    "status": "complete",
    "wall_time_seconds_including_evaluation": end - start,
    "training_epochs": 10,
    "device": "RTX 2060 SUPER",
    "purpose": "runtime and learning benchmark; not final checkpoint selection",
}, indent=2) + "\n")
' "$S0_START_EPOCH" "$S0_END_EPOCH" "$OUT/benchmark_runtime.json"
rm "$OUT/RUNNING"
date -Is > "$OUT/COMPLETED_AT"
touch "$OUT/DONE"
echo "=== S0 ten-epoch benchmark COMPLETE $(date -Is) ==="
