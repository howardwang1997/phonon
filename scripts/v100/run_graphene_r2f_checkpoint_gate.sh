#!/usr/bin/env bash
# Evaluate fixed checkpoints from the depth-3 terminal-layer fine-tune.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon}"
RUN_TAG="${RUN_TAG:-depth3_lastblock_direct}"
SEED="${SEED:-83}"
TRAIN_OUT="${TRAIN_OUT:-/data/graphene_r2f_last_layer/${RUN_TAG}_seed${SEED}}"
NAME="${NAME:-gr_r2f_${RUN_TAG}_seed${SEED}}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-1,5,10,20,40,80,120,160}"
FINAL_MODEL="$TRAIN_OUT/$NAME.model"
DATA="${DATA:-/data/graphene_r2c_loss_only}"
EVAL_DATA="${EVAL_DATA:-/data/graphene_r2c_eval}"
BASE_MODEL="${BASE_MODEL:-$ROOT/results/gr_backbone_v11/ft_graphene.model}"
BACKGROUND="${BACKGROUND:-$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml}"
CORRECTED_RESULT="${CORRECTED_RESULT:-$EVAL_DATA/corrected_result.npz}"
OUTPUT="${OUTPUT:-$TRAIN_OUT/checkpoint_development_gate.json}"

export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

while [[ ! -e "$TRAIN_OUT/TRAINING_DONE" ]]; do
    if [[ -e "$TRAIN_OUT/FAILED" ]]; then
        echo "R2F training failed before gate" >&2
        exit 3
    fi
    sleep 30
done
[[ -s "$FINAL_MODEL" ]]

for _ in $(seq 1 30); do
    GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if (( GPU_USED_MIB < 1000 )); then break; fi
    sleep 10
done
if (( GPU_USED_MIB >= 1000 )); then
    echo "GPU did not become idle after training: ${GPU_USED_MIB} MiB" >&2
    exit 4
fi

for path in "$DATA/supported9_train.xyz" \
    "$EVAL_DATA/thermal/T300/test.xyz" "$EVAL_DATA/thermal/T450/test.xyz" \
    "$EVAL_DATA/thermal/T600/test.xyz" "$EVAL_DATA/replay/val.xyz" \
    "$EVAL_DATA/operators/T300_operator.npz" "$BASE_MODEL" "$BACKGROUND" \
    "$CORRECTED_RESULT"; do
    [[ -s "$path" ]]
done

CANDIDATE_ARGS=()
for epoch in ${CHECKPOINT_EPOCHS//,/ }; do
    model="$TRAIN_OUT/checkpoint_models/epoch${epoch}.model"
    if [[ -s "$model" ]]; then
        CANDIDATE_ARGS+=(--candidate "epoch${epoch}=$model")
    fi
done
CANDIDATE_ARGS+=(--candidate "final=$FINAL_MODEL")

"$CONDA" run --no-capture-output -n "$CONDA_ENV" python \
    "$ROOT/scripts/smearing_kink/evaluate_graphene_r2c_candidates.py" \
    --base-model "$BASE_MODEL" \
    "${CANDIDATE_ARGS[@]}" \
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
echo "R2F checkpoint gate complete: $OUTPUT"
