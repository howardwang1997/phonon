#!/usr/bin/env bash
# Automatic, gated L0 queue after the S0 three-temperature force gate passes.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
S0="$ROOT/results/graphene_physics_temperature/post_p4_feasibility/S0_unified_short"
FORMAL="${FORMAL:-$S0/formal_240ep_2060_seed83_replayw16}"
OUT="$S0/L0_classical_tdep"
DATA="$ROOT/data/graphene_physical_s0"
BASE="$ROOT/results/gr_backbone_v11/ft_graphene.model"
DELTA="$FORMAL/selected_checkpoint.model"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
FREEZE="$OUT/freeze_manifest.json"
LOG="$OUT/run.log"
TARGET_PILOT=120
TARGET_FINAL=3000

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$OUT/DONE" ]]; then
    echo "physical S0 L0 queue already reached a terminal state"
    exit 0
fi
if [[ ! -e "$FORMAL/PASSED_DEVELOPMENT_FORCE_GATE" ]]; then
    echo "refusing L0: S0 development force gate has not passed"
    exit 2
fi
for path in "$DATA/manifest.json" "$FORMAL/checkpoint_sweep.json" "$DELTA" \
    "$BASE" "$BG" "$DATA/operators/T300_operator.npz" \
    "$DATA/operators/T450_operator.npz" "$DATA/operators/T600_operator.npz" \
    scripts/smearing_kink/td_phonon_friedel.py \
    scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
    scripts/smearing_kink/summarize_graphene_physical_s0_l0.py \
    scripts/smearing_kink/bootstrap_graphene_fd_conditioned_sampling.py; do
    [[ -s "$path" ]]
done

date -Is > "$OUT/STARTED_AT"
touch "$OUT/RUNNING"
L0_START_EPOCH="$(date +%s)"
L0_MONITOR_PID=""
finish() {
    L0_EXIT_CODE=$?
    trap - EXIT
    rm -f "$OUT/RUNNING"
    if [[ -n "$L0_MONITOR_PID" ]]; then
        kill "$L0_MONITOR_PID" 2>/dev/null || true
        wait "$L0_MONITOR_PID" 2>/dev/null || true
    fi
    echo "$L0_EXIT_CODE" > "$OUT/EXIT_CODE"
    if (( L0_EXIT_CODE != 0 )); then
        touch "$OUT/FAILED_INFRASTRUCTURE"
    fi
    exit "$L0_EXIT_CODE"
}
trap finish EXIT

echo "timestamp_iso,gpu_util_percent,memory_used_MiB,memory_total_MiB" > "$OUT/gpu_samples.csv"
(
    while [[ -e "$OUT/RUNNING" ]]; do
        L0_GPU_SAMPLE="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)"
        echo "$(date -Is),$L0_GPU_SAMPLE" >> "$OUT/gpu_samples.csv"
        sleep 30
    done
) &
L0_MONITOR_PID=$!

if [[ ! -s "$FREEZE" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/freeze_graphene_physical_s0_l0.py \
        --data-manifest "$DATA/manifest.json" \
        --checkpoint-sweep "$FORMAL/checkpoint_sweep.json" \
        --selected-model "$DELTA" --base-model "$BASE" --background "$BG" \
        --td-script scripts/smearing_kink/td_phonon_friedel.py \
        --recompute-script scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
        --summary-script scripts/smearing_kink/summarize_graphene_physical_s0_l0.py \
        --bootstrap-script scripts/smearing_kink/bootstrap_graphene_fd_conditioned_sampling.py \
        --output "$FREEZE"
fi

snapshot_count() {
    local path="$1"
    if [[ ! -s "$path" ]]; then
        echo 0
        return
    fi
    "$CONDA" run -n phonon python -c \
        'import numpy as np,sys; print(len(np.load(sys.argv[1],allow_pickle=False)["positions"]))' \
        "$path" | tail -1
}

wait_for_resources() {
    while true; do
        local available_kb gpu_used
        available_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
        gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
        if [[ -n "$available_kb" ]] && (( available_kb >= 52428800 )) && (( gpu_used <= 1000 )); then
            rm -f "$OUT/WAITING_STORAGE" "$OUT/WAITING_GPU"
            return
        fi
        if [[ -z "$available_kb" ]] || (( available_kb < 52428800 )); then
            touch "$OUT/WAITING_STORAGE"
            echo "waiting: RTX filesystem has less than 50 GiB free ($(date -Is))"
        fi
        if (( gpu_used > 1000 )); then
            touch "$OUT/WAITING_GPU"
            echo "waiting: another GPU task uses ${gpu_used} MiB ($(date -Is))"
        fi
        sleep 60
    done
}

settings_for_seed() {
    case "$1" in
        0|2) echo "0.5 3000 80 dt0p5" ;;
        1) echo "0.25 6000 160 dt0p25" ;;
        *) return 2 ;;
    esac
}

checkpoint_for_seed() {
    local temperature="$1" seed="$2"
    local dt equil stride dt_tag
    read -r dt equil stride dt_tag <<< "$(settings_for_seed "$seed")"
    echo "$OUT/T${temperature}/checkpoints/seed${seed}_${dt_tag}/T${temperature}"
}

run_seed() {
    local temperature="$1" seed="$2" target="$3" phase="$4"
    local dt equil stride dt_tag
    read -r dt equil stride dt_tag <<< "$(settings_for_seed "$seed")"
    local lane="$OUT/T${temperature}"
    local checkpoint_root="$lane/checkpoints/seed${seed}_${dt_tag}"
    local checkpoint="$checkpoint_root/T${temperature}"
    local snapshots="$checkpoint/snapshots.npz"
    local operator="$DATA/operators/T${temperature}_operator.npz"
    local count
    mkdir -p "$lane"
    count="$(snapshot_count "$snapshots")"
    if (( count < target )); then
        wait_for_resources
        echo "=== L0 T=$temperature seed=$seed phase=$phase snapshots=$count->$target START $(date -Is) ==="
        "$CONDA" run --no-capture-output -n phonon python \
            scripts/smearing_kink/td_phonon_friedel.py \
            --model "$BASE" --delta-model "$DELTA" --bg "$BG" \
            --operator "$operator" --friedel-tel "$temperature" --device cuda \
            --tag "graphene_physical_s0_L0_T${temperature}_seed${seed}_${phase}" \
            --temperatures "$temperature" --dt "$dt" --equil "$equil" \
            --nsnap "$target" --stride "$stride" --npoints 201 --seed "$seed" \
            --checkpoint-root "$checkpoint_root" --checkpoint-every 25 \
            --max-temperature-factor 5 --min-pair-distance 0.8 --max-force 100
    fi
    count="$(snapshot_count "$snapshots")"
    [[ "$count" == "$target" ]]
}

recompute_short() {
    local temperature="$1" checkpoint="$2" output="$3" tag="$4"
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
        --background "$BG" --checkpoint "$checkpoint" \
        --subtract-operator "$DATA/operators/T${temperature}_operator.npz" \
        --temperature "$temperature" --tag "$tag" --output "$output"
}

summarize_lane() {
    local temperature="$1" target="$2" mode="$3" suffix="$4"
    local lane="$OUT/T${temperature}"
    local checkpoints=()
    for seed in 0 1 2; do
        checkpoints+=("$(<"$lane/${suffix}_seed${seed}_checkpoint.txt")")
    done
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/summarize_graphene_physical_s0_l0.py \
        --freeze-manifest "$FREEZE" --temperature "$temperature" \
        --checkpoint "${checkpoints[0]}" --checkpoint "${checkpoints[1]}" \
        --checkpoint "${checkpoints[2]}" \
        --short-tdep "$lane/${suffix}_short_seed0.npz" \
        --short-tdep "$lane/${suffix}_short_seed1.npz" \
        --short-tdep "$lane/${suffix}_short_seed2.npz" \
        --pooled-short-tdep "$lane/${suffix}_short_pooled.npz" \
        --expected-snapshots-per-seed "$target" --mode "$mode" \
        --physical-output "$lane/${suffix}_physical_tdep.npz" \
        --output "$lane/${suffix}_acceptance.json"
}

echo "=== physical S0 automatic L0 queue START $(date -Is) ==="

# Stage 1: a reusable 120-snapshot stability pilot at all three temperatures.
if [[ ! -e "$OUT/PILOT_DONE" ]]; then
    for temperature in 450 300 600; do
        lane="$OUT/T${temperature}"
        mkdir -p "$lane"
        for seed in 0 1 2; do
            run_seed "$temperature" "$seed" "$TARGET_PILOT" pilot_n120
            checkpoint="$(checkpoint_for_seed "$temperature" "$seed")"
            printf '%s\n' "$checkpoint" > "$lane/pilot_seed${seed}_checkpoint.txt"
            recompute_short "$temperature" "$checkpoint" \
                "$lane/pilot_short_seed${seed}.npz" \
                "graphene_physical_s0_L0_T${temperature}_pilot_short_seed${seed}"
        done
        checkpoints=(
            "$(<"$lane/pilot_seed0_checkpoint.txt")"
            "$(<"$lane/pilot_seed1_checkpoint.txt")"
            "$(<"$lane/pilot_seed2_checkpoint.txt")"
        )
        "$CONDA" run --no-capture-output -n phonon python \
            scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
            --background "$BG" --checkpoint "${checkpoints[0]}" \
            --checkpoint "${checkpoints[1]}" --checkpoint "${checkpoints[2]}" \
            --subtract-operator "$DATA/operators/T${temperature}_operator.npz" \
            --temperature "$temperature" \
            --tag "graphene_physical_s0_L0_T${temperature}_pilot_short_pooled" \
            --output "$lane/pilot_short_pooled.npz"
        if ! summarize_lane "$temperature" "$TARGET_PILOT" pilot pilot; then
            touch "$lane/PILOT_STABILITY_FAILED" "$OUT/BLOCKED_PILOT_STABILITY" "$OUT/DONE"
            echo "L0 pilot failed at T=$temperature; final extension remains blocked"
            exit 0
        fi
        touch "$lane/PILOT_STABILITY_PASSED"
    done
    touch "$OUT/PILOT_DONE"
fi

# Stage 2: fixed 3000-snapshot extensions and one point/bootstrap gate per T.
for temperature in 450 300 600; do
    lane="$OUT/T${temperature}"
    if [[ -e "$lane/FINAL_GATE_PASSED" ]]; then
        continue
    fi
    for seed in 0 1 2; do
        run_seed "$temperature" "$seed" "$TARGET_FINAL" final_n3000
        checkpoint="$(checkpoint_for_seed "$temperature" "$seed")"
        printf '%s\n' "$checkpoint" > "$lane/final_seed${seed}_checkpoint.txt"
        recompute_short "$temperature" "$checkpoint" \
            "$lane/final_short_seed${seed}.npz" \
            "graphene_physical_s0_L0_T${temperature}_final_short_seed${seed}"
    done
    checkpoints=(
        "$(<"$lane/final_seed0_checkpoint.txt")"
        "$(<"$lane/final_seed1_checkpoint.txt")"
        "$(<"$lane/final_seed2_checkpoint.txt")"
    )
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
        --background "$BG" --checkpoint "${checkpoints[0]}" \
        --checkpoint "${checkpoints[1]}" --checkpoint "${checkpoints[2]}" \
        --subtract-operator "$DATA/operators/T${temperature}_operator.npz" \
        --temperature "$temperature" \
        --tag "graphene_physical_s0_L0_T${temperature}_final_short_pooled" \
        --output "$lane/final_short_pooled.npz"
    if ! summarize_lane "$temperature" "$TARGET_FINAL" final final; then
        touch "$lane/FINAL_POINT_GATE_FAILED" "$OUT/BLOCKED_FINAL_POINT_GATE" "$OUT/DONE"
        echo "L0 fixed point gate failed at T=$temperature; no extra samples will be appended"
        exit 0
    fi
    touch "$lane/FINAL_POINT_GATE_PASSED"
    if "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/bootstrap_graphene_fd_conditioned_sampling.py \
        --background "$BG" --operator "$DATA/operators/T${temperature}_operator.npz" \
        --checkpoint "0=${checkpoints[0]}/snapshots.npz" \
        --checkpoint "1=${checkpoints[1]}/snapshots.npz" \
        --checkpoint "2=${checkpoints[2]}/snapshots.npz" \
        --seed-tdep "0=$lane/final_short_seed0.npz" \
        --seed-tdep "1=$lane/final_short_seed1.npz" \
        --seed-tdep "2=$lane/final_short_seed2.npz" \
        --l0-freeze-manifest "$FREEZE" \
        --sampling-acceptance "$lane/final_acceptance.json" \
        --temperature "$temperature" --replicates 1000 \
        --random-seed "$((temperature * 1000 + 1))" --checkpoint-every 20 \
        --output "$lane/final_bootstrap_acceptance.json"; then
        touch "$lane/FINAL_BOOTSTRAP_GATE_PASSED" "$lane/FINAL_GATE_PASSED"
    else
        touch "$lane/FINAL_BOOTSTRAP_GATE_FAILED" "$OUT/BLOCKED_FINAL_BOOTSTRAP_GATE" "$OUT/DONE"
        echo "L0 bootstrap gate failed at T=$temperature; no extra samples will be appended"
        exit 0
    fi
done

L0_END_EPOCH="$(date +%s)"
"$CONDA" run -n phonon python -c '
import json, pathlib, sys
start, end = map(int, sys.argv[1:3])
pathlib.Path(sys.argv[3]).write_text(json.dumps({
    "status": "complete",
    "wall_time_seconds": end - start,
    "temperatures_K": [300, 450, 600],
    "snapshots_per_seed": 3000,
    "seeds_per_temperature": 3,
    "bootstrap_replicates_per_temperature": 1000,
}, indent=2) + "\n")
' "$L0_START_EPOCH" "$L0_END_EPOCH" "$OUT/runtime.json"
date -Is > "$OUT/COMPLETED_AT"
touch "$OUT/L0_ALL_TEMPERATURES_PASSED" "$OUT/DONE"
echo "=== physical S0 automatic L0 queue COMPLETE $(date -Is) ==="
