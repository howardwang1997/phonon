#!/usr/bin/env bash
# V100-A polls V100-B, verifies the copied bundle, then atomically releases lane A.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
REMOTE="${REMOTE:-root@100.123.220.57}"
REMOTE_SOURCE="${REMOTE_SOURCE:-/data/graphene_e0_epw_matched/k18_q9_ex1_pifroz/operator_q6_450K/cartesian}"
INPUT_ROOT="${INPUT_ROOT:-/data/graphene_e1_cross_degauss/input}"
STAGE="$INPUT_ROOT/.operator_release_staging"
BUNDLE="$INPUT_ROOT/operator_bundle"

if [[ -s "$INPUT_ROOT/OPERATOR_GATE_PASS" ]]; then
    echo "[E1-release-A] already released"
    exit 0
fi
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "test -s '$REMOTE_SOURCE/OPERATOR_GATE_PASS'"; then
    echo "[E1-release-A] remote operator gate not ready"
    exit 0
fi
mkdir -p "$STAGE" "$BUNDLE"
exec 9>"$INPUT_ROOT/.release.lock"
flock -n 9 || exit 0
scp -q -o BatchMode=yes -o ConnectTimeout=10 \
    "$REMOTE:$REMOTE_SOURCE/operator_gate_decision.json" \
    "$REMOTE:$REMOTE_SOURCE/T300_operator.npz" \
    "$REMOTE:$REMOTE_SOURCE/T450_operator.npz" \
    "$REMOTE:$REMOTE_SOURCE/T600_operator.npz" \
    "$REMOTE:$REMOTE_SOURCE/OPERATOR_GATE_PASS" "$STAGE/"
/root/miniconda3/bin/conda run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/verify_graphene_e1_operator_release.py" \
    --decision "$STAGE/operator_gate_decision.json" \
    --operator-dir "$STAGE" --output "$STAGE/verified_release.json"
for name in operator_gate_decision.json verified_release.json \
    T300_operator.npz T450_operator.npz T600_operator.npz; do
    cp -p "$STAGE/$name" "$BUNDLE/$name"
done
cp -p "$STAGE/OPERATOR_GATE_PASS" "$INPUT_ROOT/OPERATOR_GATE_PASS.tmp"
mv "$INPUT_ROOT/OPERATOR_GATE_PASS.tmp" "$INPUT_ROOT/OPERATOR_GATE_PASS"
echo "[E1-release-A] released=$(date -Is)"
