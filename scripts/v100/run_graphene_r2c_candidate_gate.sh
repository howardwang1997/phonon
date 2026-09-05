#!/usr/bin/env bash
# Wait for one R2C training run and evaluate its final model with frozen gates.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon}"
TRAIN_OUT="${TRAIN_OUT:?TRAIN_OUT is required}"
MODEL="${MODEL:?MODEL is required}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:?CANDIDATE_LABEL is required}"
DATA="${DATA:-/data/graphene_r2c_loss_only}"
EVAL_DATA="${EVAL_DATA:-/data/graphene_r2c_eval}"
BASE_MODEL="${BASE_MODEL:-$ROOT/results/gr_backbone_v11/ft_graphene.model}"
BACKGROUND="${BACKGROUND:-$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml}"
CORRECTED_RESULT="${CORRECTED_RESULT:-$EVAL_DATA/corrected_result.npz}"
OUTPUT="${OUTPUT:-$TRAIN_OUT/final_development_gate.json}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"

export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

while [[ ! -e "$TRAIN_OUT/TRAINING_DONE" ]]; do
    if [[ -e "$TRAIN_OUT/FAILED" ]]; then
        echo "training failed before gate: $TRAIN_OUT" >&2
        exit 3
    fi
    sleep "$WAIT_SECONDS"
done

for path in "$MODEL" "$DATA/supported9_train.xyz" \
    "$EVAL_DATA/thermal/T300/test.xyz" "$EVAL_DATA/thermal/T450/test.xyz" \
    "$EVAL_DATA/thermal/T600/test.xyz" "$EVAL_DATA/replay/val.xyz" \
    "$EVAL_DATA/operators/T300_operator.npz" "$BASE_MODEL" "$BACKGROUND" \
    "$CORRECTED_RESULT"; do
    [[ -s "$path" ]]
done

# TRAINING_DONE is written before the training service fully releases CUDA.
for _ in $(seq 1 20); do
    GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if (( GPU_USED_MIB < 1000 )); then
        break
    fi
    sleep 10
done
if (( GPU_USED_MIB >= 1000 )); then
    echo "GPU did not become idle after training: ${GPU_USED_MIB} MiB" >&2
    exit 4
fi

"$CONDA" run --no-capture-output -n "$CONDA_ENV" python \
    "$ROOT/scripts/smearing_kink/evaluate_graphene_r2c_candidates.py" \
    --base-model "$BASE_MODEL" \
    --candidate "$CANDIDATE_LABEL=$MODEL" \
    --support-data "$DATA/supported9_train.xyz" \
    --thermal "T300=$EVAL_DATA/thermal/T300/test.xyz" \
    --thermal "T450=$EVAL_DATA/thermal/T450/test.xyz" \
    --thermal "T600=$EVAL_DATA/thermal/T600/test.xyz" \
    --harmonic "$EVAL_DATA/replay/val.xyz" \
    --operator "$EVAL_DATA/operators/T300_operator.npz" \
    --background "$BACKGROUND" \
    --corrected-result "$CORRECTED_RESULT" \
    --device cuda --output "$OUTPUT"

date -Is > "$TRAIN_OUT/GATE_COMPLETED_AT"
touch "$TRAIN_OUT/GATE_DONE"
echo "R2C candidate gate complete: $OUTPUT"
