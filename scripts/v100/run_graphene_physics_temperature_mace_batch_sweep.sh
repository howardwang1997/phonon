#!/usr/bin/env bash
# Bounded batch-size throughput sweep using only 300/600 K development data.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
BASELINE="/data/graphene_physics_temperature/benchmarks/mace_joint_10epoch_v100b/run_seed83"
DATA="/data/graphene_physics_temperature/benchmarks/mace_joint_10epoch_v100b/joint_data"
SWEEP="${SWEEP:-/data/graphene_physics_temperature/benchmarks/mace_batch_sweep_v100b}"
LOG="$SWEEP/run.log"

cd "$ROOT"
mkdir -p "$SWEEP"
exec 9>"$SWEEP/.sweep.lock"
if ! flock -n 9; then
    echo "MACE batch-size sweep is already running"
    exit 0
fi
exec > >(tee -a "$LOG") 2>&1

if [[ -e "$SWEEP/DONE" || -e "$SWEEP/FAILED" ]]; then
    echo "MACE batch-size sweep already has a terminal marker"
    exit 0
fi

export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

for path in "$BASELINE/benchmark_summary.json" "$DATA/manifest.json" \
    "$DATA/train.xyz" "$DATA/val.xyz" "$DATA/test.xyz"; do
    [[ -s "$path" ]]
    case "$path" in
        *T450*|*P4*|*p4*)
            echo "refusing input from the frozen 450 K P4 scope: $path" >&2
            exit 64
            ;;
    esac
done

available_kb="$(df -Pk "$SWEEP" | awk 'NR==2 {print $4}')"
if [[ -z "$available_kb" || "$available_kb" -lt 52428800 ]]; then
    echo "refusing batch sweep: less than 50 GiB free" >&2
    exit 75
fi

t300_weight="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["recommended_config_weights"]["thermal_300_relative_to_600"])' \
    "$DATA/manifest.json" | tail -n 1)"

run_candidate() {
    local batch_size="$1"
    local lane="$SWEEP/batch${batch_size}"
    local name="gr_physics_proxy_batch${batch_size}_3ep_seed83"
    local gpu_samples="$lane/gpu_samples.csv"
    local resource_usage="$lane/resource_usage.txt"
    mkdir -p "$lane" "$lane/checkpoints" "$lane/logs" "$lane/results"
    if [[ -e "$lane/DONE" || -e "$lane/FAILED" || -e "$lane/TIMED_OUT" ]]; then
        echo "batch=$batch_size already has a terminal marker"
        return 0
    fi

    printf 'timestamp,utilization_gpu_percent,memory_used_MiB,power_draw_W\n' > "$gpu_samples"
    nvidia-smi \
        --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits -l 2 >> "$gpu_samples" 2>&1 &
    local monitor_pid=$!
    local start_iso start_epoch end_iso end_epoch train_rc
    start_iso="$(date -Is)"
    start_epoch="$(date +%s)"
    echo "=== batch=$batch_size START $start_iso ==="
    set +e
    timeout --signal=TERM --kill-after=60s 1200 \
        /usr/bin/time -v -o "$resource_usage" \
        "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
            --name "$name" --model MACE --num_interactions 2 \
            --hidden_irreps '32x0e+32x1o' --r_max 5.0 --num_radial_basis 8 \
            --num_cutoff_basis 5 --correlation 2 \
            --train_file "$DATA/train.xyz" --valid_file "$DATA/val.xyz" \
            --test_file "$DATA/test.xyz" \
            --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
            --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
            --config_type_weights "{'physical_fd_thermal_300K':${t300_weight},'physical_fd_thermal_600K':1.0,'physical_fd_thermal_300K_validation':${t300_weight},'physical_fd_thermal_600K_validation':1.0,'physical_fd_thermal_300K_test':${t300_weight},'physical_fd_thermal_600K_test':1.0,'v11_fc_distillation_replay':12.0,'v11_fc_distillation_replay_validation':12.0,'Default':12.0}" \
            --batch_size "$batch_size" --valid_batch_size 1 \
            --max_num_epochs 3 --patience 10 --eval_interval 3 --lr 0.001 \
            --weight_decay 1.0e-8 --ema --ema_decay 0.99 \
            --default_dtype float32 --device cuda --seed 83 --save_cpu \
            --keep_checkpoints \
            --model_dir "$lane" --checkpoints_dir "$lane/checkpoints" \
            --log_dir "$lane/logs" --results_dir "$lane/results"
    train_rc=$?
    set -e
    end_epoch="$(date +%s)"
    end_iso="$(date -Is)"
    if kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    printf 'batch_size\ttrain_exit_code\tstart_iso\tend_iso\telapsed_seconds\n%s\t%s\t%s\t%s\t%s\n' \
        "$batch_size" "$train_rc" "$start_iso" "$end_iso" "$((end_epoch - start_epoch))" \
        > "$lane/status.tsv"
    case "$train_rc" in
        0) touch "$lane/DONE" ;;
        124|137) touch "$lane/TIMED_OUT" ;;
        *) touch "$lane/FAILED" ;;
    esac
    echo "=== batch=$batch_size END rc=$train_rc $end_iso ==="
    [[ "$train_rc" -eq 0 ]]
}

overall_rc=0
for batch_size in 4 8 16 32; do
    run_candidate "$batch_size" || overall_rc=1
done

git_head="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/summarize_graphene_mace_batch_sweep.py \
    --sweep-root "$SWEEP" --baseline-dir "$BASELINE" \
    --data-manifest "$DATA/manifest.json" --output "$SWEEP/batch_sweep_summary.json" \
    --git-head "$git_head"

summary_status="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$SWEEP/batch_sweep_summary.json" | tail -n 1)"
if [[ "$summary_status" == complete && "$overall_rc" -eq 0 ]]; then
    touch "$SWEEP/DONE"
else
    touch "$SWEEP/FAILED"
    exit 1
fi
echo "=== MACE batch-size sweep COMPLETE $(date -Is) ==="
