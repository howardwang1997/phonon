#!/usr/bin/env bash
# Analyze the completed q18/q9 nine-point gate and emit one atomic decision.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
SOURCE="${SOURCE:-/data/graphene_e0_epw_matched/k18_q9_ex1_pifroz}"
INPUT_ROOT="${INPUT_ROOT:-/data/graphene_e0_epw_matched/e0_postprocess_inputs}"
ANALYSIS="${ANALYSIS:-$SOURCE/analysis}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"

test -s "$SOURCE/DONE"
test -s "$SOURCE/gate_450K/epw.out"
test -s "$SOURCE/gate_450K/specfun_sup.phon"
test -s "$INPUT_ROOT/graphene_FD450_LINE_dfpt.csv"
test -s "$INPUT_ROOT/graphene_sc6_dg0.020_phonopy.yaml"

if [[ -s "$ANALYSIS/spectral_gate_decision.json" ]] && {
    [[ -s "$ANALYSIS/SPECTRAL_GATE_PASS" ]] || [[ -s "$ANALYSIS/SPECTRAL_GATE_FAIL" ]]
}; then
    echo "[e0-postprocess] decision already complete"
    exit 0
fi

mkdir -p "$ANALYSIS"
"$CONDA" run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/analyze_graphene_e0_epw_static.py" \
    --epw-output "$SOURCE/gate_450K/epw.out" \
    --static-self-energy "$SOURCE/gate_450K/specfun_sup.phon" \
    --dfpt-450 "$INPUT_ROOT/graphene_FD450_LINE_dfpt.csv" \
    --static-reference "$INPUT_ROOT/graphene_sc6_dg0.020_phonopy.yaml" \
    --output-dir "$ANALYSIS"
"$CONDA" run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/decide_graphene_e0_spectral_gate.py" \
    --summary "$ANALYSIS/static_pilot_summary.json" \
    --output-dir "$ANALYSIS"
echo "[e0-postprocess] complete=$(date -Is)"
