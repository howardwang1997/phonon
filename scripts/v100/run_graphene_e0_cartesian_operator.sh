#!/usr/bin/env bash
# Build all three q6 Cartesian operators and materialize the aggregate E1 gate.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
SOURCE="${SOURCE:-/data/graphene_e0_epw_matched/k18_q9_ex1_pifroz}"
REFERENCE="${REFERENCE:-/data/graphene_e0_epw_matched/operator_reference/T450_long_range_operator.npz}"
MATDYN="$SOURCE/operator_q6_450K/matdyn"
OUTPUT="$SOURCE/operator_q6_450K/cartesian"
EPW_MATRIX="$MATDYN/epw_q6_dynamical_matrices.npz"
EPW_MATRIX_AUDIT="$MATDYN/epw_q6_dynamical_matrix_audit.json"

test -s "$SOURCE/full_q6_multitemp/DONE"
test -s "$REFERENCE"
mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/.lock"
if ! flock -n 9; then
    echo "[e0-operator] another operator process is already running"
    exit 0
fi
if [[ -s "$OUTPUT/OPERATOR_GATE_PASS" ]]; then
    echo "[e0-operator] aggregate gate already passed"
    exit 0
fi

if [[ ! -s "$MATDYN/MATDYN_Q6_DONE" ]]; then
    /usr/bin/bash "$ROOT/scripts/v100/run_graphene_e0_matdyn_q6.sh"
fi
/root/miniconda3/bin/conda run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/extract_graphene_epw_dynamical_matrices.py" \
    --epwdata "$SOURCE/epwdata.fmt" --crystal "$SOURCE/crystal.fmt" \
    --decay-dynmat "$SOURCE/decay.dynmat" \
    --qpoints "$SOURCE/full_q6_450K/qpoints_full.dat" \
    --static-self-energy "$SOURCE/full_q6_450K/specfun_sup.phon" \
    --nqc1 9 --nqc2 9 --nqc3 1 --ngrid 6 \
    --output "$EPW_MATRIX" --audit "$EPW_MATRIX_AUDIT"
test -s "$EPW_MATRIX"

for temperature in 300 450 600; do
    if [[ ! -s "$OUTPUT/T${temperature}_OPERATOR_PASS" ]]; then
        /root/miniconda3/bin/conda run --no-capture-output -n phonon python \
            "$ROOT/scripts/smearing_kink/build_graphene_e0_cartesian_operator.py" \
            --qpoints "$SOURCE/full_q6_${temperature}K/qpoints_full.dat" \
            --static-self-energy "$SOURCE/full_q6_${temperature}K/specfun_sup.phon" \
            --dynamical-matrices "$EPW_MATRIX" \
            --dynamical-matrix-audit "$EPW_MATRIX_AUDIT" \
            --full-q-audit "$SOURCE/full_q6_${temperature}K/audit/full_q_data_audit.json" \
            --reference-operator "$REFERENCE" \
            --temperature-K "$temperature" --ngrid 6 --output-dir "$OUTPUT"
    fi
done

/root/miniconda3/bin/conda run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/finalize_graphene_e0_operator_gate.py" \
    --operator-dir "$OUTPUT"
test -s "$OUTPUT/OPERATOR_GATE_PASS"
echo "[e0-operator] aggregate Cartesian gate passed=$(date -Is)"
