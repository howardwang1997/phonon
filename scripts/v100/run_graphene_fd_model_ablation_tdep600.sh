#!/usr/bin/env bash
# Run the selected 600 K three-seed TDEP lane on V100-B and relay it to 2060.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
OUT="$ROOT/results/graphene_fd_model_ablation/T600_TDEP"
TD="$ROOT/results/td_phonon"
REMOTE="howardwang@100.105.21.7"
REMOTE_ROOT=/home/howardwang/phonon
REMOTE_ABLATION="$REMOTE_ROOT/results/graphene_fd_model_ablation"
SELECTION="$OUT/selection.json"
LOG="$OUT/run.log"
SSH=(env -u LD_LIBRARY_PATH ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
SCP=(env -u LD_LIBRARY_PATH scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)

cd "$ROOT"
mkdir -p "$OUT" "$TD"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

while ! "${SSH[@]}" "$REMOTE" "test -s '$REMOTE_ABLATION/selection.json'"; do
    echo "waiting for 2060 model selection: $(date -Is)"
    sleep 60
done
"${SCP[@]}" "$REMOTE:$REMOTE_ABLATION/selection.json" "$SELECTION.partial"
mv "$SELECTION.partial" "$SELECTION"

passes="$($CONDA run -n phonon python -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["passes_both_temperature_gates"]))' \
    "$SELECTION" | tail -1)"
if [[ "$passes" != 1 ]]; then
    touch "$OUT/SKIPPED_FORCE_GATE"
    echo "force/replay selection failed; 600 K TDEP not started"
    exit 0
fi

label="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["by_temperature"]["600"]["selected"]["label"])' \
    "$SELECTION" | tail -1)"
case "$label" in
    replay12_600) variant=replay12 ;;
    thermal_only_600) variant=thermal_only ;;
    continued_600) variant=continued ;;
    *) echo "unsupported selected label: $label"; exit 2 ;;
esac
MODEL="$ROOT/results/graphene_fd_model_ablation/T600/$variant/gr_fd600_${variant}.model"
[[ -s "$MODEL" ]]

relay_file() {
    local source="$1" destination="$2"
    "${SSH[@]}" "$REMOTE" "mkdir -p '$(dirname "$destination")'"
    "${SCP[@]}" "$source" "$REMOTE:$destination.partial"
    "${SSH[@]}" "$REMOTE" "mv '$destination.partial' '$destination'"
}

run_seed() {
    local seed="$1" tag="graphene_v11_fd600_ablation_short_range_seed${seed}"
    [[ -s "$TD/td_${tag}.npz" && -s "$TD/td_${tag}.csv" ]] && return 0
    local specification dt equil stride suffix
    for specification in "0.5:3000:80:dt0p5" "0.25:6000:160:dt0p25"; do
        IFS=: read -r dt equil stride suffix <<< "$specification"
        if "$CONDA" run --no-capture-output -n phonon python scripts/td_phonon.py \
            --structure data/td_phonon/graphene.xyz --model "$MODEL" --device cuda \
            --tag "$tag" --outdir results/td_phonon --temperatures 600 \
            --supercell 6,6,1 --no-relax --a 2.4600000087 --dt "$dt" \
            --equil "$equil" --nsnap 120 --stride "$stride" --npoints 201 \
            --seed "$seed" --checkpoint-root "results/td_phonon/${tag}_${suffix}_checkpoint" \
            --checkpoint-every 25 --max-temperature-factor 5 \
            --min-pair-distance 0.8 --max-force 100; then
            return 0
        fi
    done
    return 1
}

for seed in 0 1 2; do
    if ! run_seed "$seed"; then
        touch "$OUT/FAILED_STABILITY"
        relay_file "$LOG" "$REMOTE_ABLATION/T600_TDEP_run.log"
        "${SSH[@]}" "$REMOTE" "touch '$REMOTE_ABLATION/TDEP600_FAILED'"
        echo "600 K seed=$seed failed all timestep attempts"
        exit 0
    fi
    stem="td_graphene_v11_fd600_ablation_short_range_seed${seed}"
    relay_file "$TD/${stem}.npz" "$REMOTE_ROOT/results/td_phonon/${stem}.npz"
    relay_file "$TD/${stem}.csv" "$REMOTE_ROOT/results/td_phonon/${stem}.csv"
done
relay_file "$LOG" "$REMOTE_ABLATION/T600_TDEP_run.log"
touch "$OUT/DONE"
"${SSH[@]}" "$REMOTE" "touch '$REMOTE_ABLATION/TDEP600_REMOTE_DONE'"
echo "=== graphene model ablation 600 K TDEP COMPLETE $(date -Is) ==="
