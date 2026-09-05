#!/usr/bin/env bash
# Evaluate every checkpoint retained by the weighted 600 K delta experiment.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
WEIGHTED="$ROOT/results/graphene_fd_delta_weighted/T600"
DATA="$ROOT/data/graphene_fd_delta_pilot/T600"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
HARMONIC="$ROOT/data/finetune_graphene8/val.xyz"
LOG="$WEIGHTED/checkpoint_sweep.log"

cd "$ROOT"
mkdir -p "$WEIGHTED"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

while [[ ! -e "$WEIGHTED/DONE" || ! -e "$WEIGHTED/W12_DONE" ]]; do
    echo "waiting for weighted 600 K models including weight-12 refinement: $(date -Is)"
    sleep 60
done
if [[ -e "$WEIGHTED/CHECKPOINT_SWEEP_DONE" ]]; then
    exit 0
fi

for tag in replayw4 replayw8 replayw12 replayw16; do
    lane="$WEIGHTED/$tag"
    name="gr_fd600_delta32_${tag}"
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/evaluate_graphene_fd_delta_checkpoints.py \
        --base-model "$V11" --template-model "$lane/${name}.model" \
        --checkpoint-dir "$lane/checkpoints" \
        --thermal "$DATA/test.xyz" --harmonic "$HARMONIC" --device cuda \
        --scope "development only: reused three-structure thermal gate" \
        --output "$lane/checkpoint_sweep.json" \
        --selected-model "$lane/${name}_checkpoint_selected.model"
done

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/select_graphene_fd_delta_checkpoint_sweeps.py \
    --candidate "replayw4=$WEIGHTED/replayw4/checkpoint_sweep.json" \
    --candidate "replayw8=$WEIGHTED/replayw8/checkpoint_sweep.json" \
    --candidate "replayw12=$WEIGHTED/replayw12/checkpoint_sweep.json" \
    --candidate "replayw16=$WEIGHTED/replayw16/checkpoint_sweep.json" \
    --output "$WEIGHTED/checkpoint_selection.json"
selected_model="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["selected_model"])' \
    "$WEIGHTED/checkpoint_selection.json" | tail -1)"
selected_tag="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["tag"])' \
    "$WEIGHTED/checkpoint_selection.json" | tail -1)"
selection_status="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$WEIGHTED/checkpoint_selection.json" | tail -1)"
if [[ "$selection_status" == development_candidate_found ]]; then
    cp -p "$selected_model" "$WEIGHTED/selected_checkpoint.model"
    printf '%s\n' "$selected_tag" > "$WEIGHTED/selected_checkpoint_tag.txt"
    touch "$WEIGHTED/CHECKPOINT_SELECTION_DONE"
else
    cp -p "$selected_model" "$WEIGHTED/provisional_selected_checkpoint.model"
    printf '%s\n' "$selected_tag" > "$WEIGHTED/provisional_selected_checkpoint_tag.txt"
    touch "$WEIGHTED/FINAL_RETRAIN_REQUIRED"
fi
touch "$WEIGHTED/CHECKPOINT_SWEEP_DONE"
echo "=== weighted 600 K checkpoint sweep COMPLETE $(date -Is) ==="
