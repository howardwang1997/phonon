#!/usr/bin/env bash
# Calibrate the finite-temperature q-space correction after the 300 K static control.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DFPT="$ROOT/results/graphene_physical_fd_dfpt/campaigns/FD300_LINE/graphene_FD300_LINE_dfpt.csv"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
PILOT="$ROOT/results/graphene_fd_delta_pilot/T300"
TDEP="$ROOT/results/graphene_fd_delta_pilot/T300_TDEP"
TD="$ROOT/results/td_phonon"
POOLED="$TD/td_graphene_v11_fd300_delta_pilot_short_pooled360.npz"
DFT="$TD/graphene_physical_fd_dft_300K_wave3.npz"
STATIC_CONTROL="$TDEP/qspace_transfer/acceptance.json"
OUT="$TDEP/qspace_calibrated"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$OUT/DONE" ]]; then
    exit 0
fi
while [[ ! -e "$TDEP/qspace_transfer/DONE" ]]; do
    echo "waiting for 300 K static-transfer control: $(date -Is)"
    sleep 60
done
for path in "$DFPT" "$BG" "$POOLED" "$DFT" "$STATIC_CONTROL" \
    "$PILOT/delta32/gate_selection.json" \
    "$TD/td_graphene_v11_fd300_delta_pilot_short_seed0.npz" \
    "$TD/td_graphene_v11_fd300_delta_pilot_short_seed1.npz" \
    "$TD/td_graphene_v11_fd300_delta_pilot_short_seed2.npz"; do
    test -s "$path"
done

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_fd_finite_temperature_calibrated_qspace.py \
    --temperature 300 --degauss 0.0019000869 \
    --dfpt-line "$DFPT" --background "$BG" \
    --thermal-short-tdep "$POOLED" --dft-tdep "$DFT" \
    --thermal-seed "$TD/td_graphene_v11_fd300_delta_pilot_short_seed0.npz" \
    --thermal-seed "$TD/td_graphene_v11_fd300_delta_pilot_short_seed1.npz" \
    --thermal-seed "$TD/td_graphene_v11_fd300_delta_pilot_short_seed2.npz" \
    --force-selection "$PILOT/delta32/gate_selection.json" \
    --static-transfer-acceptance "$STATIC_CONTROL" \
    --scope "300 K development finite-temperature calibration; not an independent force or lattice-temperature holdout" \
    --output-dir "$OUT"
if "$CONDA" run -n phonon python -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["passes_all_force_seed_calibrated_qspace_gates"] else 1)' \
    "$OUT/acceptance.json"; then
    touch "$OUT/PASSED_CALIBRATED_QSPACE_GATE"
else
    touch "$OUT/BLOCKED_CALIBRATED_QSPACE_GATE"
fi
touch "$OUT/DONE"
echo "=== 300 K finite-temperature calibrated q-space COMPLETE $(date -Is) ==="
