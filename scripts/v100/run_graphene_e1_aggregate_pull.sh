#!/usr/bin/env bash
# V100-A pulls the completed B lane and evaluates the E1 residual gate.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
REMOTE="${REMOTE:-root@100.123.220.57}"
REMOTE_B="${REMOTE_B:-/data/graphene_e1_cross_degauss/B}"
BASE="${BASE:-/data/graphene_e1_cross_degauss}"
INPUT="$BASE/input"
LANE_A="$BASE/A"
LANE_B="$BASE/B_pulled"
OUTPUT="$BASE/analysis"

if [[ -s "$OUTPUT/E1_RESIDUAL_GATE_PASS" || \
      -s "$OUTPUT/E1_RESIDUAL_GATE_FAIL" ]]; then
    echo "[E1-aggregate] decision already complete"
    exit 0
fi
if [[ ! -s "$LANE_A/DONE" || ! -s "$LANE_A/summary.json" ]]; then
    echo "[E1-aggregate] lane A is not complete"
    exit 0
fi
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "test -s '$REMOTE_B/DONE' -a -s '$REMOTE_B/summary.json'"; then
    echo "[E1-aggregate] lane B is not complete"
    exit 0
fi

mkdir -p "$LANE_B/labels" "$OUTPUT"
exec 9>"$BASE/.aggregate.lock"
flock -n 9 || exit 0
scp -q -o BatchMode=yes -o ConnectTimeout=10 \
    "$REMOTE:$REMOTE_B/summary.json" "$REMOTE:$REMOTE_B/DONE" "$LANE_B/"
scp -q -o BatchMode=yes -o ConnectTimeout=10 \
    "$REMOTE:$REMOTE_B/labels/"'*.npz' "$LANE_B/labels/"
test "$(find "$LANE_B/labels" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 15

/root/miniconda3/bin/conda run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/analyze_graphene_e1_cross_degauss.py" \
    --manifest "$INPUT/e1_manifest.json" --configs "$INPUT/e1_configs.xyz" \
    --lane-a "$LANE_A" --lane-b "$LANE_B" \
    --operator-dir "$INPUT/operator_bundle" --output-dir "$OUTPUT"
test -s "$OUTPUT/E1_RESIDUAL_GATE_PASS" -o -s "$OUTPUT/E1_RESIDUAL_GATE_FAIL"
echo "[E1-aggregate] complete=$(date -Is)"
