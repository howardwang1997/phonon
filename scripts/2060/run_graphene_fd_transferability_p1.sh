#!/usr/bin/env bash
# Cross-evaluate the frozen 300 and 600 K delta models on both endpoint datasets.
set -euo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
OUT="$ROOT/results/graphene_fd_transferability/P1_cross_temperature"
V11="$ROOT/results/gr_backbone_v11/ft_graphene.model"
HARMONIC="$ROOT/data/finetune_graphene8/val.xyz"
THERMAL300="$ROOT/data/graphene_fd_delta_pilot/T300/test.xyz"
THERMAL600="$ROOT/data/graphene_fd_delta_pilot/T600/test.xyz"
MODEL300="$ROOT/results/graphene_fd_delta_pilot/T300/delta32/gr_fd300_delta32.model"
MODEL600="$ROOT/results/graphene_fd_delta_weighted/T600/selected_checkpoint.model"
LOG="$OUT/run.log"

cd "$ROOT"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$OUT/DONE" ]]; then
    echo "P1 cross-temperature diagnostic already complete"
    exit 0
fi
for path in "$V11" "$HARMONIC" "$THERMAL300" "$THERMAL600" \
    "$MODEL300" "$MODEL600"; do
    [[ -s "$path" ]]
done

echo "=== graphene transferability P1 START $(date -Is) ==="
run_candidate() {
    local label="$1"
    local model="$2"
    local output="$OUT/${label}_metrics.json"
    if [[ -s "$output" ]] && "$CONDA" run -n phonon python -c '
import hashlib,json,sys
def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda: handle.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()
d=json.load(open(sys.argv[1]))
raise SystemExit(0 if d["delta_model_sha256"] == digest(sys.argv[2])
                 and d["base_model_sha256"] == digest(sys.argv[3]) else 1)
' "$output" "$model" "$V11"; then
        echo "reuse checksum-matched $output"
        return 0
    fi
    "$CONDA" run --no-capture-output -n phonon python \
        scripts/smearing_kink/evaluate_graphene_fd_delta_model.py \
        --dataset "thermal300=$THERMAL300" \
        --dataset "thermal600=$THERMAL600" \
        --dataset "harmonic=$HARMONIC" \
        --base-model "$V11" --delta-model "$model" --device cuda \
        --output "$output"
}

run_candidate delta300 "$MODEL300"
run_candidate delta600_weighted "$MODEL600"
"$CONDA" run --no-capture-output -n phonon python \
    scripts/smearing_kink/select_graphene_fd_cross_temperature_models.py \
    --candidate "delta300=$OUT/delta300_metrics.json" \
    --candidate "delta600_weighted=$OUT/delta600_weighted_metrics.json" \
    --output "$OUT/cross_temperature_force_matrix.json"

status="$($CONDA run -n phonon python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$OUT/cross_temperature_force_matrix.json" | tail -1)"
if [[ "$status" == existing_model_found ]]; then
    touch "$OUT/EXISTING_MODEL_FOUND"
else
    touch "$OUT/JOINT_MODEL_REQUIRED"
fi
touch "$OUT/DONE"
echo "=== graphene transferability P1 COMPLETE status=$status $(date -Is) ==="
