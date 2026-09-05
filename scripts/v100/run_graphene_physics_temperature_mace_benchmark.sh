#!/usr/bin/env bash
# Bounded GPU throughput benchmark using only 300/600 K development data.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
SOURCE="$ROOT/data/graphene_fd_delta_pilot"
SOURCE_MANIFEST="$SOURCE/manifest.json"
BENCH_ROOT="${BENCH_ROOT:-/data/graphene_physics_temperature/benchmarks/mace_joint_10epoch_v100b}"
DATA="$BENCH_ROOT/joint_data"
OUT="$BENCH_ROOT/run_seed83"
LOG="$OUT/run.log"
GPU_SAMPLES="$OUT/gpu_samples.csv"
RESOURCE_USAGE="$OUT/resource_usage.txt"
NAME="gr_physics_joint_proxy_10ep_seed83"

cd "$ROOT"
mkdir -p "$OUT" "$OUT/checkpoints" "$OUT/logs" "$OUT/results"
exec 9>"$BENCH_ROOT/.benchmark.lock"
if ! flock -n 9; then
    echo "MACE throughput benchmark is already running"
    exit 0
fi
exec > >(tee -a "$LOG") 2>&1

if [[ -e "$OUT/DONE" || -e "$OUT/TIMED_OUT" || -e "$OUT/FAILED" ]]; then
    echo "MACE throughput benchmark already has a terminal marker"
    exit 0
fi

export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

for path in "$SOURCE_MANIFEST" \
    "$SOURCE/T300/train.xyz" "$SOURCE/T300/val.xyz" "$SOURCE/T300/test.xyz" \
    "$SOURCE/T600/train.xyz" "$SOURCE/T600/val.xyz" "$SOURCE/T600/test.xyz"; do
    [[ -s "$path" ]]
    case "$path" in
        *T450*|*P4*|*p4*)
            echo "refusing benchmark input from the frozen 450 K P4 scope: $path" >&2
            exit 64
            ;;
    esac
done

available_kb="$(df -Pk "$BENCH_ROOT" | awk 'NR==2 {print $4}')"
if [[ -z "$available_kb" || "$available_kb" -lt 52428800 ]]; then
    echo "refusing benchmark: less than 50 GiB free" >&2
    exit 75
fi

if [[ ! -s "$DATA/manifest.json" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/prepare_graphene_fd_joint_delta_data.py \
        --input-root "$SOURCE" --source-manifest "$SOURCE_MANIFEST" --output "$DATA"
fi

t300_weight="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["recommended_config_weights"]["thermal_300_relative_to_600"])' \
    "$DATA/manifest.json" | tail -n 1)"
echo "joint proxy config weights: T300=$t300_weight T600=1.0"

restart=()
if compgen -G "$OUT/checkpoints/${NAME}_run-*_epoch-*.pt" >/dev/null; then
    restart+=(--restart_latest)
fi

printf 'timestamp,utilization_gpu_percent,memory_used_MiB,power_draw_W\n' > "$GPU_SAMPLES"
nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits -l 10 >> "$GPU_SAMPLES" 2>&1 &
monitor_pid=$!
cleanup_monitor() {
    if kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup_monitor EXIT

start_iso="$(date -Is)"
start_epoch="$(date +%s)"
git_head="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
echo "=== MACE joint-development 10-epoch benchmark START $start_iso ==="
set +e
timeout --signal=TERM --kill-after=120s 7200 \
    /usr/bin/time -v -o "$RESOURCE_USAGE" \
    "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
        --name "$NAME" --model MACE --num_interactions 2 \
        --hidden_irreps '32x0e+32x1o' --r_max 5.0 --num_radial_basis 8 \
        --num_cutoff_basis 5 --correlation 2 \
        --train_file "$DATA/train.xyz" --valid_file "$DATA/val.xyz" \
        --test_file "$DATA/test.xyz" \
        --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
        --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
        --config_type_weights "{'physical_fd_thermal_300K':${t300_weight},'physical_fd_thermal_600K':1.0,'physical_fd_thermal_300K_validation':${t300_weight},'physical_fd_thermal_600K_validation':1.0,'physical_fd_thermal_300K_test':${t300_weight},'physical_fd_thermal_600K_test':1.0,'v11_fc_distillation_replay':12.0,'v11_fc_distillation_replay_validation':12.0,'Default':12.0}" \
        --batch_size 1 --valid_batch_size 1 --max_num_epochs 10 \
        --patience 20 --eval_interval 5 --lr 0.001 \
        --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
        --default_dtype float32 --device cuda --seed 83 --save_cpu \
        --keep_checkpoints --save_all_checkpoints \
        --model_dir "$OUT" --checkpoints_dir "$OUT/checkpoints" \
        --log_dir "$OUT/logs" --results_dir "$OUT/results" \
        "${restart[@]}"
train_rc=$?
set -e
end_epoch="$(date +%s)"
end_iso="$(date -Is)"
cleanup_monitor
trap - EXIT

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/summarize_graphene_mace_benchmark.py \
    --output-dir "$OUT" --joint-manifest "$DATA/manifest.json" \
    --source-manifest "$SOURCE_MANIFEST" --gpu-samples "$GPU_SAMPLES" \
    --resource-usage "$RESOURCE_USAGE" --run-log "$LOG" \
    --start-iso "$start_iso" --end-iso "$end_iso" \
    --elapsed-seconds "$((end_epoch - start_epoch))" \
    --train-exit-code "$train_rc" --git-head "$git_head" --configured-epochs 10

case "$train_rc" in
    0)
        touch "$OUT/DONE"
        ;;
    124|137)
        touch "$OUT/TIMED_OUT"
        ;;
    *)
        touch "$OUT/FAILED"
        ;;
esac
echo "=== MACE benchmark END rc=$train_rc $end_iso ==="
if [[ "$train_rc" -eq 124 || "$train_rc" -eq 137 ]]; then
    exit 0
fi
exit "$train_rc"
