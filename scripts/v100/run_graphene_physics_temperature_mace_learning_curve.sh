#!/usr/bin/env bash
# Continuous 240-epoch MACE learning curve on P4-isolated development data.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
DATA="/data/graphene_physics_temperature/benchmarks/mace_joint_10epoch_v100b/joint_data"
CONTROL="/data/graphene_physics_temperature/benchmarks/mace_convergence_30ep_v100b/convergence_summary.json"
OUT="${OUT:-/data/graphene_physics_temperature/benchmarks/mace_learning_curve_240ep_v100b}"
NAME="gr_physics_joint_batch1_curve240_seed83"
LANE="$OUT/train"
LOG="$OUT/run.log"
GPU_SAMPLES="$OUT/gpu_samples.csv"

cd "$ROOT"
mkdir -p "$OUT" "$LANE/checkpoints" "$LANE/logs" "$LANE/results" \
    "$OUT/models" "$OUT/evaluations"
exec 9>"$OUT/.learning_curve.lock"
if ! flock -n 9; then
    echo "MACE learning curve is already running"
    exit 0
fi
exec > >(tee -a "$LOG") 2>&1
if [[ -e "$OUT/DONE" ]]; then
    echo "MACE learning curve already complete"
    exit 0
fi
if [[ -e "$OUT/FAILED" ]]; then
    mv "$OUT/FAILED" "$OUT/FAILED.previous.$(date +%s)"
fi

export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

for path in "$DATA/manifest.json" "$DATA/train.xyz" "$DATA/val.xyz" \
    "$DATA/test.xyz" "$CONTROL"; do
    [[ -s "$path" ]]
    case "$path" in
        *T450*|*P4*|*p4*)
            echo "refusing input from the frozen 450 K P4 scope: $path" >&2
            exit 64
            ;;
    esac
done
"$CONDA" run -n phonon python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="complete"; assert d["target_leakage"]["p4_450K_inputs_read"] is False' \
    "$CONTROL"
available_kb="$(df -Pk "$OUT" | awk 'NR==2 {print $4}')"
if [[ -z "$available_kb" || "$available_kb" -lt 52428800 ]]; then
    echo "refusing learning curve: less than 50 GiB free" >&2
    exit 75
fi
t300_weight="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["recommended_config_weights"]["thermal_300_relative_to_600"])' \
    "$DATA/manifest.json" | tail -n 1)"
restart_args=()
if find "$LANE/checkpoints" -type f -name '*.pt' -print -quit | grep -q .; then
    restart_args+=(--restart_latest)
    echo "resuming from the latest MACE checkpoint"
fi

printf 'timestamp,utilization_gpu_percent,memory_used_MiB,power_draw_W\n' > "$GPU_SAMPLES"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits -l 2 >> "$GPU_SAMPLES" 2>&1 &
monitor_pid=$!
cleanup() {
    if kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_iso="$(date -Is)"
start_epoch="$(date +%s)"
echo "=== continuous MACE 240-epoch learning curve START $start_iso ==="
set +e
timeout --signal=TERM --kill-after=60s 14400 \
    /usr/bin/time -v -o "$OUT/resource_usage.txt" \
    "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
        --name "$NAME" --model MACE --num_interactions 2 \
        --hidden_irreps '32x0e+32x1o' --r_max 5.0 --num_radial_basis 8 \
        --num_cutoff_basis 5 --correlation 2 \
        --train_file "$DATA/train.xyz" --valid_file "$DATA/val.xyz" \
        --test_file "$DATA/test.xyz" \
        --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
        --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
        --config_type_weights "{'physical_fd_thermal_300K':${t300_weight},'physical_fd_thermal_600K':1.0,'physical_fd_thermal_300K_validation':${t300_weight},'physical_fd_thermal_600K_validation':1.0,'physical_fd_thermal_300K_test':${t300_weight},'physical_fd_thermal_600K_test':1.0,'v11_fc_distillation_replay':12.0,'v11_fc_distillation_replay_validation':12.0,'Default':12.0}" \
        --batch_size 1 --valid_batch_size 1 --max_num_epochs 241 \
        --patience 1000 --eval_interval 5 --lr 0.001 --weight_decay 1.0e-8 \
        --ema --ema_decay 0.99 --default_dtype float32 --device cuda \
        --seed 83 --save_cpu --keep_checkpoints --save_all_checkpoints \
        --model_dir "$LANE" --checkpoints_dir "$LANE/checkpoints" \
        --log_dir "$LANE/logs" --results_dir "$LANE/results" \
        "${restart_args[@]}"
train_rc=$?
set -e
end_epoch="$(date +%s)"
end_iso="$(date -Is)"
printf 'train_exit_code\tstart_iso\tend_iso\telapsed_seconds\n%s\t%s\t%s\t%s\n' \
    "$train_rc" "$start_iso" "$end_iso" "$((end_epoch - start_epoch))" \
    > "$OUT/status.tsv"
if [[ "$train_rc" -ne 0 ]]; then
    touch "$OUT/FAILED"
    exit "$train_rc"
fi

template="$LANE/checkpoints/${NAME}_run-83.model"
for epoch in 60 120 240; do
    checkpoint="$LANE/checkpoints/${NAME}_run-83_epoch-${epoch}.pt"
    model="$OUT/models/epoch$(printf '%03d' "$epoch").model"
    evaluation="$OUT/evaluations/epoch$(printf '%03d' "$epoch").json"
    [[ -s "$template" && -s "$checkpoint" ]]
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/materialize_mace_checkpoint.py \
        --template-model "$template" --checkpoint "$checkpoint" --output "$model"
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/evaluate_graphene_mace_development_proxy.py \
        --model "$model" --dataset "val=$DATA/val.xyz" \
        --dataset "test=$DATA/test.xyz" --device cuda --output "$evaluation"
done

git_head="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/summarize_graphene_mace_learning_curve.py \
    --root "$OUT" --control-summary "$CONTROL" \
    --data-manifest "$DATA/manifest.json" --output "$OUT/learning_curve_summary.json" \
    --git-head "$git_head"
summary_status="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$OUT/learning_curve_summary.json" | tail -n 1)"
if [[ "$summary_status" == complete ]]; then
    touch "$OUT/DONE"
else
    touch "$OUT/FAILED"
    exit 1
fi
echo "=== continuous MACE 240-epoch learning curve COMPLETE $(date -Is) ==="
