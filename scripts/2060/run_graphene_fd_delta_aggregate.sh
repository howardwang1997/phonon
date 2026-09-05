#!/usr/bin/env bash
# Aggregate frozen-v11 delta-model gates and the two parallel TDEP lanes.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
PILOT="$ROOT/results/graphene_fd_delta_pilot"
FINAL="$ROOT/results/graphene_fd_residual_finetune"
TD="$ROOT/results/td_phonon"
LOG="$PILOT/aggregate.log"

cd "$ROOT"
mkdir -p "$PILOT" "$FINAL"
exec > >(tee -a "$LOG") 2>&1
touch "$PILOT/PROVISIONAL_OPERATOR_PILOT"

while [[ ! -e "$PILOT/T300/DONE" || ! -e "$PILOT/T600/REMOTE_DONE" ]]; do
    echo "waiting for 300/600 K frozen-v11 delta-model gates: $(date -Is)"
    sleep 60
done
if [[ -e "$PILOT/T300/BLOCKED_FORCE_GATE" || -e "$PILOT/T600/BLOCKED_FORCE_GATE" ]]; then
    touch "$PILOT/BLOCKED_FORCE_GATE" "$PILOT/DONE"
    echo "at least one frozen-v11 delta lane failed the independent force gate"
    exit 0
fi

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/combine_graphene_fd_delta_selections.py \
    --metrics-300 "$PILOT/T300/selected_gate_metrics.json" \
    --metrics-600 "$PILOT/T600/selected_gate_metrics.json" \
    --selection-300 "$PILOT/T300/selection.json" \
    --selection-600 "$PILOT/T600/selection.json" \
    --force-output "$PILOT/force_validation.json" \
    --stack-output "$PILOT/model_stack.json"

while [[ ! -e "$PILOT/T300_TDEP/DONE" \
      || ( ! -e "$PILOT/TDEP600_REMOTE_DONE" \
           && ! -e "$PILOT/TDEP600_FAILED" \
           && ! -e "$PILOT/TDEP600_SKIPPED" ) ]]; do
    echo "waiting for parallel 300/600 K full-force TDEP: $(date -Is)"
    sleep 60
done
if [[ -e "$PILOT/T300_TDEP/FAILED_STABILITY" || -e "$PILOT/TDEP600_FAILED" ]]; then
    touch "$PILOT/BLOCKED_STABILITY" "$PILOT/DONE"
    echo "at least one full-force delta trajectory failed its health checks"
    exit 0
fi
if [[ -e "$PILOT/T300_TDEP/SKIPPED_FORCE_GATE" || -e "$PILOT/TDEP600_SKIPPED" ]]; then
    touch "$PILOT/BLOCKED_FORCE_GATE" "$PILOT/DONE"
    exit 0
fi

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_physical_fd_tdep.py \
    --root "$TD" --force-validation "$PILOT/force_validation.json" \
    --wave wave3 --previous-wave wave2 --model-tag delta_pilot_full \
    --final-wave --high-symmetry-threshold 50 \
    --output "$PILOT/acceptance.json"
passed="$($CONDA run -n phonon python -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["passes_short_range_gate"]))' \
    "$PILOT/acceptance.json" | tail -1)"
if [[ "$passed" == 1 ]]; then
    touch "$PILOT/PASSED_PILOT_GATE"
else
    touch "$PILOT/BLOCKED_TDEP_GATE"
fi
echo delta_pilot_full > "$PILOT/selected_tag.txt"
touch "$PILOT/DONE"
echo "provisional operator pilot complete; final release waits for dense-DFPT calibration"
echo "=== graphene frozen-v11 delta aggregate COMPLETE $(date -Is) ==="
