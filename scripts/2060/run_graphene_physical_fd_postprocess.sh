#!/usr/bin/env bash
# Final automatic validation queue: wait for the accepted thermal short-range
# model and dense direct DFPT data, fit the q-space correction, then build the
# degauss-to-zero reference when FD0_SCAN arrives.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
DFPT="$ROOT/results/graphene_physical_fd_dfpt/campaigns"
WAVE2="$ROOT/results/graphene_fd_thermal_finetune_wave2/acceptance.json"
WAVE3="$ROOT/results/graphene_fd_thermal_finetune_wave3/acceptance.json"
ABLATION_V2="$ROOT/results/graphene_fd_model_ablation_v2"
RESIDUAL="$ROOT/results/graphene_fd_residual_finetune"
OUT="$ROOT/results/graphene_physical_fd_long_range"
ZERO_OUT="$ROOT/results/graphene_physical_fd0"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT" "$ZERO_OUT"
exec > >(tee -a "$LOG") 2>&1
rm -f "$OUT/BLOCKED_SHORT_RANGE_GATE"
echo "=== graphene physical-FD final postprocess queue start $(date -Is) ==="
while [[ ! -s "$WAVE2" ]]; do
    echo "waiting for wave2 acceptance: $(date -Is)"
    sleep 300
done

wave2_pass="$($CONDA run -n phonon python -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["passes_short_range_gate"]))' \
    "$WAVE2" | tail -1)"
short_range_args=(--wave2-acceptance "$WAVE2")
if [[ "$wave2_pass" != 1 ]]; then
    while [[ ! -s "$WAVE3" ]]; do
        echo "wave2 failed its fixed gate; waiting for conditional wave3: $(date -Is)"
        sleep 300
    done
    wave3_pass="$($CONDA run -n phonon python -c \
        'import json,sys; print(int(json.load(open(sys.argv[1]))["passes_short_range_gate"]))' \
        "$WAVE3" | tail -1)"
    if [[ "$wave3_pass" == 1 ]]; then
        short_range_args+=(--wave3-acceptance "$WAVE3")
    else
        while [[ ! -s "$RESIDUAL/acceptance.json" && ! -e "$RESIDUAL/DONE" ]]; do
            echo "wave3 failed; waiting for long-range-subtracted residual model: $(date -Is)"
            sleep 300
        done
        if [[ ! -s "$RESIDUAL/acceptance.json" ]]; then
            touch "$OUT/BLOCKED_SHORT_RANGE_GATE"
            echo "residual-force workflow found no accepted short-range model; long-range fitting remains blocked"
            exit 0
        fi
        residual_pass="$($CONDA run -n phonon python -c \
            'import json,sys; print(int(json.load(open(sys.argv[1]))["passes_short_range_gate"]))' \
            "$RESIDUAL/acceptance.json" | tail -1)"
        if [[ "$residual_pass" != 1 ]]; then
            touch "$OUT/BLOCKED_SHORT_RANGE_GATE"
            echo "residual-force model failed the short-range gate; long-range fitting remains blocked"
            exit 0
        fi
        residual_tag="$(<"$RESIDUAL/selected_tag.txt")"
        short_range_args=(
            --short-range-acceptance "$RESIDUAL/acceptance.json"
            --short-range-tag "$residual_tag"
        )
    fi
fi
rm -f "$OUT/BLOCKED_SHORT_RANGE_GATE"

while [[ ! -s "$DFPT/FD300_LINE/graphene_FD300_LINE_dfpt.csv" \
      || ! -s "$DFPT/FD600_LINE/graphene_FD600_LINE_dfpt.csv" ]]; do
    echo "short-range gate passed; waiting for 300/600 K dense DFPT lines: $(date -Is)"
    sleep 300
done

if [[ ! -e "$OUT/DONE" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/evaluate_graphene_physical_fd_long_range.py \
        --dfpt-300 "$DFPT/FD300_LINE/graphene_FD300_LINE_dfpt.csv" \
        --dfpt-600 "$DFPT/FD600_LINE/graphene_FD600_LINE_dfpt.csv" \
        --td-root results/td_phonon "${short_range_args[@]}" --output-dir "$OUT"
    touch "$OUT/DONE"
fi

while [[ ! -s "$DFPT/FD0_SCAN/graphene_FD0_SCAN_dfpt.csv" ]]; do
    echo "long-range validation complete; waiting for degauss-to-zero scan: $(date -Is)"
    sleep 300
done
if [[ ! -e "$ZERO_OUT/DONE" ]]; then
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/analyze_graphene_fd0_extrapolation.py \
        --fd0 "$DFPT/FD0_SCAN/graphene_FD0_SCAN_dfpt.csv" \
        --fd300 "$DFPT/FD300_LINE/graphene_FD300_LINE_dfpt.csv" \
        --fd600 "$DFPT/FD600_LINE/graphene_FD600_LINE_dfpt.csv" \
        --output-dir "$ZERO_OUT"
    touch "$ZERO_OUT/DONE"
fi
touch "$OUT/POSTPROCESS_DONE"
echo "=== graphene physical-FD final postprocess COMPLETE $(date -Is) ==="
