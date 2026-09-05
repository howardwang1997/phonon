#!/usr/bin/env bash
# Apply the frozen full q-space LR transfer to the existing 300 K delta lane.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DFPT="$ROOT/results/graphene_physical_fd_dfpt/campaigns/FD300_LINE/graphene_FD300_LINE_dfpt.csv"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
PILOT="$ROOT/results/graphene_fd_delta_pilot/T300"
OPERATOR="$ROOT/data/graphene_fd_delta_pilot/T300/long_range_operator.npz"
TDEP="$ROOT/results/graphene_fd_delta_pilot/T300_TDEP"
TD="$ROOT/results/td_phonon"
STATIC="$TDEP/static_test/static_short_fc2.npz"
POOLED="$TD/td_graphene_v11_fd300_delta_pilot_short_pooled360.npz"
DFT="$TD/graphene_physical_fd_dft_300K_wave3.npz"
OUT="$TDEP/qspace_transfer"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

for seed in 0 1 2; do
    checkpoint="$TD/graphene_v11_fd300_delta_pilot_full_short_range_seed${seed}_dt1_checkpoint/T300"
    short_seed="$TD/td_graphene_v11_fd300_delta_pilot_short_seed${seed}.npz"
    [[ -s "$checkpoint/snapshots.npz" ]]
    if [[ ! -s "$short_seed" ]]; then
        "$CONDA" run --no-capture-output -n phonon python \
            scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py \
            --background "$BG" --checkpoint "$checkpoint" \
            --subtract-operator "$OPERATOR" --temperature 300 \
            --tag "graphene_v11_fd300_delta_pilot_short_seed${seed}" \
            --output "$short_seed"
    fi
done
while [[ ! -s "$DFPT" ]]; do
    echo "waiting for complete FD300_LINE relay: $(date -Is)"
    sleep 60
done
if [[ -e "$OUT/DONE" ]]; then
    exit 0
fi
inputs=(
    "$BG" "$STATIC" "$POOLED" "$DFT" "$OPERATOR"
    "$PILOT/delta32/gate_selection.json"
    "$TD/td_graphene_v11_fd300_delta_pilot_short_seed0.npz"
    "$TD/td_graphene_v11_fd300_delta_pilot_short_seed1.npz"
    "$TD/td_graphene_v11_fd300_delta_pilot_short_seed2.npz"
)
for path in "${inputs[@]}"; do [[ -s "$path" ]]; done

"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/evaluate_graphene_fd_finite_temperature_qspace.py \
    --temperature 300 --degauss 0.0019000869 \
    --dfpt-line "$DFPT" --background "$BG" \
    --static-short-fc2 "$STATIC" --thermal-short-tdep "$POOLED" \
    --dft-tdep "$DFT" \
    --thermal-seed "$TD/td_graphene_v11_fd300_delta_pilot_short_seed0.npz" \
    --thermal-seed "$TD/td_graphene_v11_fd300_delta_pilot_short_seed1.npz" \
    --thermal-seed "$TD/td_graphene_v11_fd300_delta_pilot_short_seed2.npz" \
    --force-selection "$PILOT/delta32/gate_selection.json" \
    --scope "300 K development force gate plus second-lattice-temperature q-space constraint; not an independent force holdout" \
    --output-dir "$OUT"
if "$CONDA" run -n phonon python -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["passes_all_force_seed_qspace_gates"] else 1)' \
    "$OUT/acceptance.json"; then
    touch "$OUT/PASSED_FORCE_SEED_QSPACE_GATE"
else
    touch "$OUT/BLOCKED_SEED_OR_QSPACE_GATE"
fi
touch "$OUT/DONE"
echo "=== 300 K full q-space LR validation COMPLETE $(date -Is) ==="
