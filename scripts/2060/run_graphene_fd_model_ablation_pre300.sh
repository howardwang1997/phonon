#!/usr/bin/env bash
# Use the idle 2060 to screen and validate the completed 300 K lane early.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DATA="$ROOT/data/graphene_fd_model_ablation"
OUT="$ROOT/results/graphene_fd_model_ablation"
TD="$ROOT/results/td_phonon"
LOG="$OUT/pre300.log"

cd "$ROOT"
mkdir -p "$OUT" "$TD"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while [[ ! -e "$OUT/T300/DONE" ]]; do
    echo "waiting for 300 K model variants: $(date -Is)"
    sleep 60
done
if [[ -e "$OUT/PRE300_DONE" || -e "$OUT/PRE300_BLOCKED_FORCE_GATE" ]]; then
    exit 0
fi

model_args=(
    --model "v11=$ROOT/results/gr_backbone_v11/ft_graphene.model"
    --model "balanced_300=$ROOT/results/graphene_fd_thermal_finetune_wave3/T300/gr_v11_fd300_wave3.model"
)
for variant in replay12 thermal_only continued; do
    model_args+=(--model "${variant}_300=$OUT/T300/${variant}/gr_fd300_${variant}.model")
done

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_model_variants.py \
    --dataset "thermal300=$DATA/replay0/T300/test.xyz" \
    --dataset "harmonic=data/finetune_graphene8/val.xyz" \
    "${model_args[@]}" --device cuda --output "$OUT/variant_force_metrics_300.json"

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/select_graphene_model_variants.py \
    --metrics "$OUT/variant_force_metrics_300.json" \
    --candidate-300 replay12_300 --candidate-300 thermal_only_300 \
    --candidate-300 continued_300 --output "$OUT/selection_300.json"

passes="$($CONDA run -n phonon python -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["passes_all_requested_temperature_gates"]))' \
    "$OUT/selection_300.json" | tail -1)"
if [[ "$passes" != 1 ]]; then
    touch "$OUT/PRE300_BLOCKED_FORCE_GATE"
    echo "no 300 K candidate passed the thermal-force and harmonic-replay gates"
    exit 0
fi
model="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["by_temperature"]["300"]["selected"]["path"])' \
    "$OUT/selection_300.json" | tail -1)"

run_seed() {
    local seed="$1" tag="graphene_v11_fd300_ablation_short_range_seed${seed}"
    [[ -s "$TD/td_${tag}.npz" && -s "$TD/td_${tag}.csv" ]] && return 0
    "$CONDA" run --no-capture-output -n phonon python scripts/td_phonon.py \
        --structure data/td_phonon/graphene.xyz --model "$model" --device cuda \
        --tag "$tag" --outdir results/td_phonon --temperatures 300 \
        --supercell 6,6,1 --no-relax --a 2.4600000087 --dt 1.0 \
        --equil 1500 --nsnap 120 --stride 40 --npoints 201 --seed "$seed" \
        --checkpoint-root "results/td_phonon/${tag}_dt1_checkpoint" \
        --checkpoint-every 25 --max-temperature-factor 5 \
        --min-pair-distance 0.8 --max-force 100
}

for seed in 0 1 2; do
    if ! run_seed "$seed"; then
        touch "$OUT/PRE300_STABILITY_FAILED"
        echo "300 K seed=$seed failed its trajectory health checks"
        exit 0
    fi
done
touch "$OUT/PRE300_DONE"
echo "=== graphene model ablation 300 K prevalidation COMPLETE $(date -Is) ==="
