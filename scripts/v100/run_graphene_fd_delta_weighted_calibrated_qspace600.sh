#!/usr/bin/env bash
# Follow-up after the preregistered static-transfer control: calibrate LR at 600 K.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
PHONON_ENV="${PHONON_ENV:-phonon}"
OUT="$ROOT/results/graphene_fd_delta_weighted/T600_TDEP"
QSPACE="$OUT/qspace_calibrated"
TD="$ROOT/results/td_phonon"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
DFT="$TD/graphene_physical_fd_dft_600K_wave3.npz"
DFPT_LINE="$ROOT/results/graphene_physical_fd_dfpt/campaigns/FD600_LINE/graphene_FD600_LINE_dfpt.csv"
POOLED="$TD/td_graphene_v11_fd600_delta_weighted_short_pooled360.npz"
FORCE="$ROOT/results/graphene_fd_delta_weighted/holdout600/gate_selection.json"
STATIC_CONTROL="$OUT/acceptance.json"
REMOTE="howardwang@100.105.21.7"
REMOTE_ROOT=/home/howardwang/phonon
REMOTE_OUT="$REMOTE_ROOT/results/graphene_fd_delta_weighted/T600_TDEP"
SSH=(env -u LD_LIBRARY_PATH ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
SCP=(env -u LD_LIBRARY_PATH scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)

cd "$ROOT"
mkdir -p "$QSPACE"
exec > >(tee -a "$OUT/calibrated_run.log") 2>&1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$QSPACE/DONE" ]]; then
    exit 0
fi
while [[ ! -e "$OUT/DONE" ]]; do
    echo "waiting for static-transfer control and three-seed aggregate: $(date -Is)"
    sleep 60
done
for path in "$DFPT_LINE" "$BG" "$POOLED" "$DFT" "$FORCE" "$STATIC_CONTROL" \
    "$TD/td_graphene_v11_fd600_delta_weighted_short_seed0.npz" \
    "$TD/td_graphene_v11_fd600_delta_weighted_short_seed1.npz" \
    "$TD/td_graphene_v11_fd600_delta_weighted_short_seed2.npz"; do
    test -s "$path"
done

"$CONDA" run --no-capture-output -n "$PHONON_ENV" python \
    scripts/smearing_kink/evaluate_graphene_fd_finite_temperature_calibrated_qspace.py \
    --temperature 600 --degauss 0.0038001738 \
    --dfpt-line "$DFPT_LINE" --background "$BG" \
    --thermal-short-tdep "$POOLED" --dft-tdep "$DFT" \
    --thermal-seed "$TD/td_graphene_v11_fd600_delta_weighted_short_seed0.npz" \
    --thermal-seed "$TD/td_graphene_v11_fd600_delta_weighted_short_seed1.npz" \
    --thermal-seed "$TD/td_graphene_v11_fd600_delta_weighted_short_seed2.npz" \
    --force-selection "$FORCE" --static-transfer-acceptance "$STATIC_CONTROL" \
    --scope "600 K finite-temperature-calibrated q-space correction after the unchanged static-transfer control failed" \
    --output-dir "$QSPACE"
cp -p "$QSPACE/acceptance.json" "$OUT/calibrated_acceptance.json"
if "$CONDA" run -n "$PHONON_ENV" python -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["passes_all_force_seed_calibrated_qspace_gates"] else 1)' \
    "$OUT/calibrated_acceptance.json"; then
    touch "$OUT/PASSED_CALIBRATED_QSPACE_GATE"
else
    touch "$OUT/BLOCKED_CALIBRATED_QSPACE_GATE"
fi

relay() {
    local local_path="$1" remote_path="$2"
    "${SSH[@]}" "$REMOTE" "mkdir -p '$(dirname "$remote_path")'"
    "${SCP[@]}" "$local_path" "$REMOTE:$remote_path.partial"
    "${SSH[@]}" "$REMOTE" "mv '$remote_path.partial' '$remote_path'"
}
relay "$OUT/calibrated_acceptance.json" "$REMOTE_OUT/calibrated_acceptance.json"
relay "$QSPACE/finite_temperature_calibrated_predictions.csv" \
    "$REMOTE_OUT/finite_temperature_calibrated_predictions.csv"
relay "$QSPACE/finite_temperature_calibrated_comparison.png" \
    "$REMOTE_OUT/finite_temperature_calibrated_comparison.png"
relay "$OUT/calibrated_run.log" "$REMOTE_OUT/calibrated_run.log"
touch "$QSPACE/DONE"
"${SSH[@]}" "$REMOTE" "touch '$REMOTE_OUT/CALIBRATED_REMOTE_DONE'"
echo "=== weighted-delta 600 K calibrated q-space COMPLETE $(date -Is) ==="
