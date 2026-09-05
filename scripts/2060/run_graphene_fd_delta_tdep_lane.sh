#!/usr/bin/env bash
# Three-seed full-force TDEP: frozen v11 + delta MACE + fixed harmonic LR.
set -euo pipefail

TEMPERATURE="${1:?usage: run_graphene_fd_delta_tdep_lane.sh 300|600}"
case "$TEMPERATURE" in 300|600) ;; *) exit 2 ;; esac
ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
MODEL_OUT="$ROOT/results/graphene_fd_delta_pilot/T${TEMPERATURE}"
OUT="$ROOT/results/graphene_fd_delta_pilot/T${TEMPERATURE}_TDEP"
TD="$ROOT/results/td_phonon"
DATA="$ROOT/data/graphene_fd_delta_pilot/T${TEMPERATURE}"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT" "$TD"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

while [[ ! -e "$MODEL_OUT/DONE" ]]; do
    echo "waiting for T=$TEMPERATURE frozen-v11 delta-model gate: $(date -Is)"
    sleep 60
done
if [[ ! -e "$MODEL_OUT/CANDIDATE_PASSED" ]]; then
    touch "$OUT/SKIPPED_FORCE_GATE" "$OUT/DONE"
    echo "delta model failed force/retention gate; TDEP skipped"
else
    variant=$(<"$MODEL_OUT/selected_variant.txt")
    DELTA="$MODEL_OUT/$variant/gr_fd${TEMPERATURE}_${variant}.model"
    OPERATOR="$DATA/long_range_operator.npz"
    for path in "$V11" "$DELTA" "$OPERATOR" "$BG"; do [[ -s "$path" ]]; done

    run_seed() {
        local seed="$1" tag="graphene_v11_fd${TEMPERATURE}_delta_pilot_full_short_range_seed${seed}"
        [[ -s "$TD/td_${tag}.npz" && -s "$TD/td_${tag}.csv" ]] && return 0
        local specifications=()
        if [[ "$TEMPERATURE" == 300 ]]; then
            specifications=("1.0:1500:40:dt1" "0.5:3000:80:dt0p5")
        else
            specifications=("0.5:3000:80:dt0p5" "0.25:6000:160:dt0p25")
        fi
        local specification dt equil stride suffix
        for specification in "${specifications[@]}"; do
            IFS=: read -r dt equil stride suffix <<< "$specification"
            if "$CONDA" run --no-capture-output -n phonon python \
                scripts/smearing_kink/td_phonon_friedel.py \
                --model "$V11" --delta-model "$DELTA" --bg "$BG" \
                --operator "$OPERATOR" --device cuda --tag "$tag" \
                --temperatures "$TEMPERATURE" --dt "$dt" --equil "$equil" \
                --nsnap 120 --stride "$stride" --npoints 201 --seed "$seed" \
                --checkpoint-root "results/td_phonon/${tag}_${suffix}_checkpoint" \
                --checkpoint-every 25 --max-temperature-factor 5 \
                --min-pair-distance 0.8 --max-force 100; then
                return 0
            fi
        done
        return 1
    }

    failed=0
    for seed in 0 1 2; do
        if ! run_seed "$seed"; then
            failed=1
            echo "T=$TEMPERATURE full-force TDEP seed=$seed failed both timesteps"
            break
        fi
    done
    if [[ "$failed" == 1 ]]; then
        touch "$OUT/FAILED_STABILITY"
    fi
    touch "$OUT/DONE"
fi

if [[ "$TEMPERATURE" == 600 ]]; then
    REMOTE="howardwang@100.105.21.7"
    REMOTE_ROOT=/home/howardwang/phonon
    REMOTE_OUT="$REMOTE_ROOT/results/graphene_fd_delta_pilot"
    SSH=(env -u LD_LIBRARY_PATH ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
    SCP=(env -u LD_LIBRARY_PATH scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
    "${SSH[@]}" "$REMOTE" "mkdir -p '$REMOTE_OUT' '$REMOTE_ROOT/results/td_phonon'"
    if [[ -e "$OUT/FAILED_STABILITY" ]]; then
        "${SSH[@]}" "$REMOTE" "touch '$REMOTE_OUT/TDEP600_FAILED'"
    elif [[ -e "$OUT/SKIPPED_FORCE_GATE" ]]; then
        "${SSH[@]}" "$REMOTE" "touch '$REMOTE_OUT/TDEP600_SKIPPED'"
    else
        for seed in 0 1 2; do
            stem="td_graphene_v11_fd600_delta_pilot_full_short_range_seed${seed}"
            for suffix in npz csv; do
                "${SCP[@]}" "$TD/${stem}.${suffix}" \
                    "$REMOTE:$REMOTE_ROOT/results/td_phonon/${stem}.${suffix}.partial"
                "${SSH[@]}" "$REMOTE" \
                    "mv '$REMOTE_ROOT/results/td_phonon/${stem}.${suffix}.partial' '$REMOTE_ROOT/results/td_phonon/${stem}.${suffix}'"
            done
        done
        "${SSH[@]}" "$REMOTE" "touch '$REMOTE_OUT/TDEP600_REMOTE_DONE'"
    fi
    "${SCP[@]}" "$LOG" "$REMOTE:$REMOTE_OUT/T600_TDEP_run.log.partial"
    "${SSH[@]}" "$REMOTE" "mv '$REMOTE_OUT/T600_TDEP_run.log.partial' '$REMOTE_OUT/T600_TDEP_run.log'"
fi
echo "=== graphene full-force delta TDEP T=$TEMPERATURE COMPLETE $(date -Is) ==="
