#!/usr/bin/env bash
# Wait for the remote B0 EPW calculation and run the full matrix/shape gate.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
DENSE_ROOT="${DENSE_ROOT:-/data/graphene_k_cusp_b0/dense_k29_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/graphene_k_cusp_b0/analysis_dense_k29}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-21600}"

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.lock"
if ! flock -n 9; then
    echo "[cusp-b0-post] another postprocessor is running"
    exit 0
fi
if [[ -s "$OUTPUT_DIR/DONE" ]]; then
    echo "[cusp-b0-post] already complete"
    exit 0
fi

waited=0
while [[ ! -s "$DENSE_ROOT/DONE" ]]; do
    sleep 60
    waited=$((waited + 60))
    if (( waited >= WAIT_TIMEOUT_SECONDS )); then
        echo "[cusp-b0-post] wait timeout after ${waited}s" >&2
        exit 3
    fi
done

date -Is > "$OUTPUT_DIR/STARTED_AT"
"$CONDA" run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/analyze_graphene_k_cusp_b0_dense.py" \
    --dense-root "$DENSE_ROOT" --output-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/analysis.log.partial" 2>&1
mv "$OUTPUT_DIR/analysis.log.partial" "$OUTPUT_DIR/analysis.log"
test -s "$OUTPUT_DIR/b0_dense_summary.json"
"$CONDA" run --no-capture-output -n phonon python -c '
import json,sys
payload=json.load(open(sys.argv[1]))
assert payload["status"] == "passed_quantitative_dense", payload["status"]
' "$OUTPUT_DIR/b0_dense_summary.json"
date -Is > "$OUTPUT_DIR/COMPLETED_AT"
cp "$OUTPUT_DIR/COMPLETED_AT" "$OUTPUT_DIR/DONE.tmp"
mv "$OUTPUT_DIR/DONE.tmp" "$OUTPUT_DIR/DONE"
echo "[cusp-b0-post] complete=$(cat "$OUTPUT_DIR/COMPLETED_AT")"
