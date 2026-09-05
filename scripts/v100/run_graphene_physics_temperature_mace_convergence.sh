#!/usr/bin/env bash
# Development-only speed/quality convergence check for fixed MACE batch/LR pairs.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
DATA="/data/graphene_physics_temperature/benchmarks/mace_joint_10epoch_v100b/joint_data"
OUT="${OUT:-/data/graphene_physics_temperature/benchmarks/mace_convergence_30ep_v100b}"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT"
exec 9>"$OUT/.convergence.lock"
if ! flock -n 9; then
    echo "MACE convergence benchmark is already running"
    exit 0
fi
exec > >(tee -a "$LOG") 2>&1
if [[ -e "$OUT/DONE" || -e "$OUT/FAILED" ]]; then
    echo "MACE convergence benchmark already has a terminal marker"
    exit 0
fi

export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

for path in "$DATA/manifest.json" "$DATA/train.xyz" "$DATA/val.xyz" "$DATA/test.xyz"; do
    [[ -s "$path" ]]
    case "$path" in
        *T450*|*P4*|*p4*)
            echo "refusing input from the frozen 450 K P4 scope: $path" >&2
            exit 64
            ;;
    esac
done
available_kb="$(df -Pk "$OUT" | awk 'NR==2 {print $4}')"
if [[ -z "$available_kb" || "$available_kb" -lt 52428800 ]]; then
    echo "refusing convergence benchmark: less than 50 GiB free" >&2
    exit 75
fi
t300_weight="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["recommended_config_weights"]["thermal_300_relative_to_600"])' \
    "$DATA/manifest.json" | tail -n 1)"

run_candidate() {
    local tag="$1"
    local batch_size="$2"
    local learning_rate="$3"
    local lane="$OUT/$tag"
    local name="gr_${tag}_30ep_seed83"
    local model="$lane/${name}.model"
    local gpu_samples="$lane/gpu_samples.csv"
    mkdir -p "$lane" "$lane/checkpoints" "$lane/logs" "$lane/results"
    if [[ -e "$lane/DONE" || -e "$lane/FAILED" || -e "$lane/TIMED_OUT" ]]; then
        echo "$tag already has a terminal marker"
        return 0
    fi
    printf 'timestamp,utilization_gpu_percent,memory_used_MiB,power_draw_W\n' > "$gpu_samples"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits -l 2 >> "$gpu_samples" 2>&1 &
    local monitor_pid=$!
    local start_iso start_epoch end_iso end_epoch train_rc
    start_iso="$(date -Is)"
    start_epoch="$(date +%s)"
    echo "=== $tag batch=$batch_size lr=$learning_rate START $start_iso ==="
    set +e
    timeout --signal=TERM --kill-after=60s 1800 \
        /usr/bin/time -v -o "$lane/resource_usage.txt" \
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
            --max_num_epochs 30 --patience 40 --eval_interval 5 \
            --lr "$learning_rate" --weight_decay 1.0e-8 \
            --ema --ema_decay 0.99 --default_dtype float32 --device cuda \
            --seed 83 --save_cpu --keep_checkpoints \
            --model_dir "$lane" --checkpoints_dir "$lane/checkpoints" \
            --log_dir "$lane/logs" --results_dir "$lane/results"
    train_rc=$?
    set -e
    end_epoch="$(date +%s)"
    end_iso="$(date -Is)"
    printf 'batch_size\ttrain_exit_code\tstart_iso\tend_iso\telapsed_seconds\n%s\t%s\t%s\t%s\t%s\n' \
        "$batch_size" "$train_rc" "$start_iso" "$end_iso" "$((end_epoch - start_epoch))" \
        > "$lane/status.tsv"
    if [[ "$train_rc" -eq 0 && -s "$model" ]]; then
        set +e
        "$CONDA" run --no-capture-output -n phonon python \
            scripts/smearing_kink/evaluate_graphene_mace_development_proxy.py \
            --model "$model" --dataset "val=$DATA/val.xyz" \
            --dataset "test=$DATA/test.xyz" --device cuda \
            --output "$lane/development_evaluation.json"
        eval_rc=$?
        set -e
    else
        eval_rc=1
    fi
    if kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if [[ "$train_rc" -eq 0 && "$eval_rc" -eq 0 ]]; then
        touch "$lane/DONE"
    elif [[ "$train_rc" -eq 124 || "$train_rc" -eq 137 ]]; then
        touch "$lane/TIMED_OUT"
    else
        touch "$lane/FAILED"
    fi
    echo "=== $tag END train_rc=$train_rc eval_rc=$eval_rc $(date -Is) ==="
    [[ "$train_rc" -eq 0 && "$eval_rc" -eq 0 ]]
}

overall_rc=0
run_candidate batch1_lr0p001 1 0.001 || overall_rc=1
run_candidate batch8_lrsqrt 8 0.00282842712474619 || overall_rc=1
run_candidate batch16_lrsqrt 16 0.004 || overall_rc=1
run_candidate batch32_lrsqrt 32 0.00565685424949238 || overall_rc=1

git_head="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/summarize_graphene_mace_convergence_benchmark.py \
    --root "$OUT" --data-manifest "$DATA/manifest.json" \
    --output "$OUT/convergence_summary.json" --git-head "$git_head"
summary_status="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$OUT/convergence_summary.json" | tail -n 1)"
if [[ "$summary_status" == complete && "$overall_rc" -eq 0 ]]; then
    touch "$OUT/DONE"
else
    touch "$OUT/FAILED"
    exit 1
fi
echo "=== MACE convergence benchmark COMPLETE $(date -Is) ==="
