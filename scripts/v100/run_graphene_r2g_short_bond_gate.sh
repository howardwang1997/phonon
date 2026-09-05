#!/usr/bin/env bash
# Evaluate all fixed checkpoints from one compact short-bond expert run.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon}"
RUN_TAG="${RUN_TAG:-compact_c2_nonlinear}"
SEED="${SEED:-83}"
TRAIN_OUT="${TRAIN_OUT:-/data/graphene_r2g_short_bond/${RUN_TAG}_seed${SEED}}"
NAME="gr_r2g_${RUN_TAG}_seed${SEED}"
FINAL_EXPERT="$TRAIN_OUT/$NAME.pt"
DATA="${DATA:-/data/graphene_r2f_joint_support_harmonic}"
EVAL_DATA="${EVAL_DATA:-/data/graphene_r2c_eval}"
BASE_MODEL="${BASE_MODEL:-$ROOT/results/gr_backbone_v11/ft_graphene.model}"
SHORT_BASE="${SHORT_BASE:-/data/graphene_r2f_last_layer/depth3_lastblock_direct_seed83/gr_r2f_depth3_lastblock_direct_seed83.model}"
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
        echo "R2G training failed before gate" >&2
        exit 3
    fi
    sleep 30
done
for path in "$FINAL_EXPERT" "$DATA/supported9_train.xyz" \
    "$EVAL_DATA/thermal/T300/test.xyz" "$EVAL_DATA/thermal/T450/test.xyz" \
    "$EVAL_DATA/thermal/T600/test.xyz" "$EVAL_DATA/replay/val.xyz" \
    "$EVAL_DATA/operators/T300_operator.npz" "$BASE_MODEL" "$SHORT_BASE" \
    "$BACKGROUND" "$CORRECTED_RESULT"; do
    [[ -s "$path" ]]
done

for _ in $(seq 1 30); do
    GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if (( GPU_USED_MIB < 1000 )); then break; fi
    sleep 10
done
if (( GPU_USED_MIB >= 1000 )); then
    echo "GPU did not become idle after training: ${GPU_USED_MIB} MiB" >&2
    exit 4
fi

CANDIDATE_ARGS=()
for expert in "$TRAIN_OUT"/checkpoint_experts/epoch*.pt; do
    [[ -s "$expert" ]]
    epoch="$(basename "$expert" .pt)"
    CANDIDATE_ARGS+=(--expert "${epoch}=$expert")
done
CANDIDATE_ARGS+=(--expert "final=$FINAL_EXPERT")

"$CONDA" run --no-capture-output -n "$CONDA_ENV" python \
    "$ROOT/scripts/smearing_kink/evaluate_graphene_r2g_short_bond_expert.py" \
    --base-model "$BASE_MODEL" --short-base-model "$SHORT_BASE" \
    "${CANDIDATE_ARGS[@]}" \
    --support-data "$DATA/supported9_train.xyz" \
    --thermal "T300=$EVAL_DATA/thermal/T300/test.xyz" \
    --thermal "T450=$EVAL_DATA/thermal/T450/test.xyz" \
    --thermal "T600=$EVAL_DATA/thermal/T600/test.xyz" \
    --harmonic "$EVAL_DATA/replay/val.xyz" \
    --operator "$EVAL_DATA/operators/T300_operator.npz" \
    --background "$BACKGROUND" --corrected-result "$CORRECTED_RESULT" \
    --device cuda --output "$OUTPUT"

date -Is > "$TRAIN_OUT/GATE_COMPLETED_AT"
touch "$TRAIN_OUT/GATE_DONE"
echo "R2G checkpoint gate complete: $OUTPUT"
