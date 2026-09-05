#!/usr/bin/env bash
# Verify the B-local operator bundle and atomically release E1 lane B.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
SOURCE="${SOURCE:-/data/graphene_e0_epw_matched/k18_q9_ex1_pifroz/operator_q6_450K/cartesian}"
INPUT_ROOT="${INPUT_ROOT:-/data/graphene_e1_cross_degauss/input}"
STAGE="$INPUT_ROOT/.operator_release_staging"
BUNDLE="$INPUT_ROOT/operator_bundle"

test -s "$SOURCE/OPERATOR_GATE_PASS"
if [[ -s "$INPUT_ROOT/OPERATOR_GATE_PASS" ]]; then
    echo "[E1-release-B] already released"
    exit 0
fi
mkdir -p "$STAGE" "$BUNDLE"
exec 9>"$INPUT_ROOT/.release.lock"
flock -n 9 || exit 0
for name in operator_gate_decision.json T300_operator.npz T450_operator.npz T600_operator.npz; do
    cp -p "$SOURCE/$name" "$STAGE/$name"
done
/root/miniconda3/bin/conda run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/verify_graphene_e1_operator_release.py" \
    --decision "$STAGE/operator_gate_decision.json" \
    --operator-dir "$STAGE" --output "$STAGE/verified_release.json"
for name in operator_gate_decision.json verified_release.json \
    T300_operator.npz T450_operator.npz T600_operator.npz; do
    cp -p "$STAGE/$name" "$BUNDLE/$name"
done
cp -p "$SOURCE/OPERATOR_GATE_PASS" "$INPUT_ROOT/OPERATOR_GATE_PASS.tmp"
mv "$INPUT_ROOT/OPERATOR_GATE_PASS.tmp" "$INPUT_ROOT/OPERATOR_GATE_PASS"
echo "[E1-release-B] released=$(date -Is)"
