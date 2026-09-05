#!/usr/bin/env bash
# One-shot evaluation on the fixed 15-structure 600 K force holdout.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
WEIGHTED="$ROOT/results/graphene_fd_delta_weighted/T600"
RAW="$ROOT/data/graphene_fd_thermal_holdout600"
OUT="$ROOT/results/graphene_fd_delta_weighted/holdout600"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
OPERATOR="$ROOT/data/graphene_fd_delta_pilot/T600/long_range_operator.npz"
HARMONIC="$ROOT/data/finetune_graphene8/val.xyz"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

while [[ ! -e "$WEIGHTED/CHECKPOINT_SELECTION_DONE" || ! -e "$RAW/RAW_READY" ]]; do
    echo "waiting for frozen checkpoint selection and raw holdout labels: $(date -Is)"
    sleep 60
done
if [[ -e "$OUT/DONE" ]]; then
    exit 0
fi
for path in "$WEIGHTED/selected_checkpoint.model" "$RAW/summary.xyz" \
    "$V11" "$OPERATOR" "$HARMONIC"; do
    [[ -s "$path" ]]
done

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/apply_graphene_fd_long_range_operator.py \
    --input "$RAW/summary.xyz" --operator "$OPERATOR" \
    --output "$OUT/holdout600_with_long_range.xyz" \
    --manifest "$OUT/holdout600_preparation.json"
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_fd_delta_model.py \
    --dataset "thermal600_holdout=$OUT/holdout600_with_long_range.xyz" \
    --dataset "harmonic=$HARMONIC" --base-model "$V11" \
    --delta-model "$WEIGHTED/selected_checkpoint.model" --device cuda \
    --output "$OUT/gate_metrics.json"
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/select_graphene_fd_delta_model.py \
    --metrics "$OUT/gate_metrics.json" --thermal-label thermal600_holdout \
    --output "$OUT/gate_selection.json"

if "$CONDA" run -n phonon python -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["passes_force_and_replay_gate"] else 1)' \
    "$OUT/gate_selection.json"; then
    touch "$OUT/INDEPENDENT_FORCE_GATE_PASSED"
else
    touch "$OUT/BLOCKED_INDEPENDENT_FORCE_GATE"
fi
touch "$OUT/DONE"
echo "=== fixed 600 K force holdout evaluation COMPLETE $(date -Is) ==="
