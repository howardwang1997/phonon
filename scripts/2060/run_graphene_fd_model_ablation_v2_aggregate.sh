#!/usr/bin/env bash
# Combine the raw-force fine-tuning diagnostics.
#
# These models are intentionally NOT eligible for TDEP or final long-range
# validation.  The physical-FD DFT labels contain the non-local Kohn force; a
# local MACE fitted directly to those total forces mixes that contribution into
# the short-range backbone.  The production path must first subtract the
# force-capable long-range operator and train on residual forces.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DATA="$ROOT/data/graphene_fd_model_ablation_v2/replay72_joint"
OUT="$ROOT/results/graphene_fd_model_ablation_v2"
TD="$ROOT/results/td_phonon"
LOG="$OUT/aggregate.log"

cd "$ROOT"
mkdir -p "$OUT" "$TD"
exec > >(tee -a "$LOG") 2>&1
touch "$OUT/DIAGNOSTIC_ONLY"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while [[ ! -e "$OUT/T300/DONE" \
      || ( ! -e "$OUT/T600/REMOTE_DONE" && ! -e "$OUT/T600/BLOCKED_FORCE_GATE" ) ]]; do
    echo "waiting for conservative 300/600 K model gates: $(date -Is)"
    sleep 60
done
if [[ -e "$OUT/T300/BLOCKED_FORCE_GATE" || -e "$OUT/T600/BLOCKED_FORCE_GATE" ]]; then
    touch "$OUT/BLOCKED_FORCE_GATE" "$OUT/DONE"
    echo "raw-force diagnostic: at least one temperature has no candidate passing both gates"
    exit 0
fi

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/combine_graphene_model_variant_selections.py \
    --selection-300 "$OUT/T300/selection.json" \
    --selection-600 "$OUT/T600/selection.json" \
    --output "$OUT/selection.json"

variant300=$(<"$OUT/T300/selected_variant.txt")
variant600=$(<"$OUT/T600/selected_variant.txt")
model300="$OUT/T300/$variant300/gr_fd300_v2_${variant300}.model"
model600="$OUT/T600/$variant600/gr_fd600_v2_${variant600}.model"
for model in "$model300" "$model600"; do [[ -s "$model" ]]; done

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_thermal_force_models.py \
    --validation-a "$DATA/T300/test.xyz" --validation-b "$DATA/T600/test.xyz" \
    --model "v11=$ROOT/results/gr_backbone_v11/ft_graphene.model" \
    --model "fd300=$model300" --model "fd600=$model600" \
    --device cuda --output "$OUT/force_validation.json"
touch "$OUT/DIAGNOSTIC_RAW_FORCE_PASSED" "$OUT/DONE"
echo "raw-force diagnostics passed their numerical force checks"
echo "TDEP intentionally skipped: production training requires long-range-force subtraction"
echo "=== graphene raw-force diagnostic aggregate COMPLETE $(date -Is) ==="
