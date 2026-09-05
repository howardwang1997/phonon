#!/usr/bin/env bash
# Recompute the 450/450 Q0 source and then 450/300 X0 with explicit atom mapping.
set -euo pipefail

ROOT="${ROOT:-/home/howardwang/phonon}"
CONDA="${CONDA:-/home/howardwang/miniconda3/bin/conda}"
WORK_ROOT="$ROOT/results/graphene_physics_temperature/post_p4_feasibility/R1_order_safe_sscha"
MANIFEST="$WORK_ROOT/order_safe_Q0_X0_freeze_manifest.json"
RUNNER="$ROOT/scripts/smearing_kink/run_graphene_physical_q0_sscha.py"
ORDER_TEST="$ROOT/scripts/smearing_kink/test_graphene_sscha_atom_order_interface.py"
Q0_OUT="$WORK_ROOT/Q0_quantum_sscha/formal_T450"
X0_OUT="$WORK_ROOT/X0_cross_development/sscha_Tlat450_Tel300"

mkdir -p "$WORK_ROOT"
exec 9>"$WORK_ROOT/.lock"
if ! flock -n 9; then
    echo "order-safe Q0/X0 is already running"
    exit 0
fi
if [[ -e "$WORK_ROOT/DONE" ]]; then
    echo "order-safe Q0/X0 is already complete"
    exit 0
fi
exec > >(tee -a "$WORK_ROOT/run.log") 2>&1

test -s "$MANIFEST"
test -s "$RUNNER"
test -s "$ORDER_TEST"
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "81e678ded024c86c6775046d91d1c92f036d3e599c053e7ca8c34796a7d9c780"
test "$(sha256sum "$RUNNER" | awk '{print $1}')" = "d4a8091602f2b7f5d03b849909b3db827280e131612ef8f260684fec37a73b79"
test "$(sha256sum "$ORDER_TEST" | awk '{print $1}')" = "9bfb5ab9481a44273615314f307f0736d3722bde65df150a90211a6f9c028cbe"

date -Is > "$WORK_ROOT/STARTED_AT"
touch "$WORK_ROOT/RUNNING"
finish() {
    local exit_code=$?
    rm -f "$WORK_ROOT/RUNNING"
    echo "$exit_code" > "$WORK_ROOT/EXIT_CODE"
    if (( exit_code != 0 )); then
        touch "$WORK_ROOT/FAILED"
    fi
}
trap finish EXIT

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/graphene-order-safe-matplotlib}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$MPLCONFIGDIR"

echo "=== atom-order interface smoke test START $(date -Is) ==="
"$CONDA" run --no-capture-output -n phonon python "$ORDER_TEST" \
    --freeze-manifest "$MANIFEST" --operator-temperature 300 \
    | tee "$WORK_ROOT/atom_order_interface_test.json"
echo "=== corrected Q0 Tlat=450 Tel=450 START $(date -Is) ==="
"$CONDA" run --no-capture-output -n phonon python "$RUNNER" \
    --freeze-manifest "$MANIFEST" --phase formal \
    --lattice-temperature 450 --operator-temperature 450 \
    --n-configs 300 --max-populations 5 --random-seed 450005 \
    --device cuda --output-dir "$Q0_OUT"
"$CONDA" run -n phonon python -c '
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "acceptance.json"
payload = json.loads(path.read_text())
if payload.get("status") != "passed" or payload.get("converged") is not True:
    raise SystemExit("corrected Q0 did not pass")
if payload.get("atom_order_interface", {}).get("status") != "validated_before_SSCHA":
    raise SystemExit("corrected Q0 lacks atom-order validation")
' "$Q0_OUT"

echo "=== corrected X0 Tlat=450 Tel=300 START $(date -Is) ==="
"$CONDA" run --no-capture-output -n phonon python "$RUNNER" \
    --freeze-manifest "$MANIFEST" --phase x0_first \
    --lattice-temperature 450 --operator-temperature 300 \
    --n-configs 300 --max-populations 5 --random-seed 453005 \
    --device cuda --output-dir "$X0_OUT"
"$CONDA" run -n phonon python -c '
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "acceptance.json"
payload = json.loads(path.read_text())
if payload.get("status") != "passed" or payload.get("converged") is not True:
    raise SystemExit("corrected X0 did not pass")
if payload.get("atom_order_interface", {}).get("status") != "validated_before_SSCHA":
    raise SystemExit("corrected X0 lacks atom-order validation")
' "$X0_OUT"

date -Is > "$WORK_ROOT/COMPLETED_AT"
touch "$WORK_ROOT/DONE"
echo "=== order-safe Q0/X0 COMPLETE $(date -Is) ==="
