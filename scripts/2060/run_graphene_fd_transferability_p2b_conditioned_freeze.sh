#!/usr/bin/env bash
# Freeze the predeclared temperature-conditioned fallback before any 450 K target.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
RESULTS="$ROOT/results/graphene_fd_transferability"
P1="$RESULTS/P1_cross_temperature/cross_temperature_force_matrix.json"
P2="$RESULTS/P2_joint_short/checkpoint_selection.json"
INVENTORY="$RESULTS/P0_sync/source_inventory.json"
BASE="$ROOT/results/gr_backbone_v11/ft_graphene.model"
DELTA300="$ROOT/results/graphene_fd_delta_pilot/T300/delta32/gr_fd300_delta32.model"
DELTA600="$ROOT/results/graphene_fd_delta_weighted/T600/selected_checkpoint.model"
HARMONIC="$ROOT/data/finetune_graphene8/val.xyz"
OP300="$ROOT/data/graphene_fd_delta_pilot/T300/long_range_operator.npz"
OP600="$ROOT/data/graphene_fd_delta_pilot/T600/long_range_operator.npz"
CONDITIONED="$RESULTS/conditioned_short"
REPLAY="$CONDITIONED/endpoint_identity_replay.json"
OP450="$CONDITIONED/T450_long_range_operator.npz"
OP450_MANIFEST="$CONDITIONED/T450_long_range_operator_manifest.json"
LAW="$RESULTS/temperature_law.json"
FREEZE="$RESULTS/freeze_manifest.json"
LOG="$RESULTS/P2b_conditioned_freeze.log"

cd "$ROOT"
mkdir -p "$RESULTS" "$CONDITIONED"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -s "$FREEZE" && -e "$RESULTS/CONDITIONED_FREEZE_DONE" ]]; then
    echo "conditioned predictor is already frozen"
    exit 0
fi
for path in "$P1" "$P2" "$INVENTORY" "$BASE" "$DELTA300" "$DELTA600" \
    "$HARMONIC" "$OP300" "$OP600" \
    "$ROOT/results/graphene_fd_delta_pilot/T300/delta32/gate_selection.json" \
    "$ROOT/results/graphene_fd_delta_pilot/T300/delta32/gate_metrics.json" \
    "$ROOT/results/graphene_fd_delta_weighted/holdout600/gate_selection.json" \
    "$ROOT/results/graphene_fd_delta_weighted/holdout600/gate_metrics.json" \
    "$ROOT/results/graphene_fd_delta_pilot/T300_TDEP/qspace_calibrated/acceptance.json" \
    "$ROOT/results/graphene_fd_delta_weighted/T600_TDEP/calibrated_acceptance.json" \
    "$ROOT/results/td_phonon/td_graphene_v11_fd300_delta_pilot_short_pooled360.npz" \
    "$ROOT/results/td_phonon/td_graphene_v11_fd600_delta_weighted_short_pooled360.npz"; do
    [[ -s "$path" ]]
done
"$CONDA" run -n phonon python -c \
    'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p["status"] == "no_joint_candidate_passed" else 1)' \
    "$P2"

echo "=== graphene transferability conditioned freeze START $(date -Is) ==="
if [[ ! -s "$OP450" || ! -s "$OP450_MANIFEST" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/interpolate_graphene_fd_long_range_operator.py \
        --operator-300 "$OP300" --operator-600 "$OP600" --temperature 450 \
        --output "$OP450" --manifest "$OP450_MANIFEST"
fi
if [[ ! -s "$REPLAY" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/evaluate_graphene_fd_conditioned_endpoints.py \
        --endpoint "300=$ROOT/data/graphene_fd_delta_pilot/T300/test.xyz" \
        --endpoint "600=$ROOT/data/graphene_fd_delta_pilot/T600/test.xyz" \
        --harmonic "$HARMONIC" --base-model "$BASE" \
        --delta-model-300 "$DELTA300" --delta-model-600 "$DELTA600" \
        --device cuda --output "$REPLAY"
fi

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/freeze_graphene_fd_conditioned_predictor.py \
    --p1-selection "$P1" --p2-selection "$P2" --source-inventory "$INVENTORY" \
    --endpoint-replay "$REPLAY" --base-model "$BASE" \
    --delta-model "300=$DELTA300" --delta-model "600=$DELTA600" \
    --force-selection "300=$ROOT/results/graphene_fd_delta_pilot/T300/delta32/gate_selection.json" \
    --force-selection "600=$ROOT/results/graphene_fd_delta_weighted/holdout600/gate_selection.json" \
    --force-metrics "300=$ROOT/results/graphene_fd_delta_pilot/T300/delta32/gate_metrics.json" \
    --force-metrics "600=$ROOT/results/graphene_fd_delta_weighted/holdout600/gate_metrics.json" \
    --endpoint-acceptance "300=$ROOT/results/graphene_fd_delta_pilot/T300_TDEP/qspace_calibrated/acceptance.json" \
    --endpoint-acceptance "600=$ROOT/results/graphene_fd_delta_weighted/T600_TDEP/calibrated_acceptance.json" \
    --pooled-tdep "300=$ROOT/results/td_phonon/td_graphene_v11_fd300_delta_pilot_short_pooled360.npz" \
    --pooled-tdep "600=$ROOT/results/td_phonon/td_graphene_v11_fd600_delta_weighted_short_pooled360.npz" \
    --operator "300=$OP300" --operator "600=$OP600" \
    --prediction-operator-manifest "$OP450_MANIFEST" \
    --plan "$ROOT/docs/GRAPHENE_FD_CONDITIONAL_FALLBACK_PLAN.md" \
    --code-path "$ROOT/scripts/smearing_kink/conditioned_mace.py" \
    --code-path "$ROOT/scripts/smearing_kink/td_phonon_friedel.py" \
    --code-path "$ROOT/scripts/smearing_kink/interpolate_graphene_fd_long_range_operator.py" \
    --code-path "$ROOT/scripts/smearing_kink/evaluate_graphene_fd_conditioned_endpoints.py" \
    --code-path "$ROOT/scripts/smearing_kink/freeze_graphene_fd_conditioned_predictor.py" \
    --code-path "$ROOT/scripts/smearing_kink/summarize_graphene_fd_conditioned_tdep.py" \
    --code-path "$ROOT/scripts/2060/run_graphene_fd_transferability_p4a.sh" \
    --code-path "$ROOT/scripts/v100/run_graphene_physical_fd_dfpt.sh" \
    --absence-path "$RESULTS/T450" \
    --absence-path "$ROOT/data/graphene_fd_transferability/T450/RAW_READY" \
    --absence-path "$ROOT/results/graphene_physical_fd_dfpt/campaigns/FD450_CONV" \
    --absence-path "$ROOT/results/graphene_physical_fd_dfpt/campaigns/FD450_LINE" \
    --prediction-temperature 450 --temperature-law-output "$LAW" --output "$FREEZE"

touch "$RESULTS/READY_FOR_450" "$RESULTS/CONDITIONED_FREEZE_DONE"
echo "=== graphene transferability conditioned freeze COMPLETE $(date -Is) ==="
