#!/usr/bin/env bash
# Evaluate fixed R2D adapter checkpoints after training completes.
set -euo pipefail

ROOT="${ROOT:-/home/howardwang/phonon}"
CONDA="${CONDA:-/home/howardwang/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon}"
RUN_TAG="${RUN_TAG:-local_mb_r3_h16_c3}"
SEED="${SEED:-83}"
TRAIN_OUT="${TRAIN_OUT:-$ROOT/results/graphene_physics_temperature/post_p4_feasibility/R2D_local_adapter/${RUN_TAG}_seed${SEED}}"
NAME="gr_r2d_${RUN_TAG}_seed${SEED}"
FINAL_MODEL="$TRAIN_OUT/$NAME.model"
OUTPUT="$TRAIN_OUT/checkpoint_development_gate.json"
DATA="${DATA:-$ROOT/data/graphene_r2c_loss_only}"
EVAL_DATA="${EVAL_DATA:-$ROOT/data/graphene_physical_s0}"
BASE_MODEL="${BASE_MODEL:-$ROOT/results/gr_backbone_v11/ft_graphene.model}"
DEPTH3_MODEL="${DEPTH3_MODEL:-$ROOT/results/graphene_physics_temperature/post_p4_feasibility/R2C_short_repair/depth3_forces_only_seed83/gr_r2c_depth3_forces_only_seed83.model}"
BACKGROUND="${BACKGROUND:-$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml}"
CORRECTED_RESULT="${CORRECTED_RESULT:-$ROOT/results/graphene_physics_temperature/post_p4_feasibility/R1_order_safe_sscha/X0_cross_development/sscha_Tlat450_Tel300/result.npz}"

export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

while [[ ! -e "$TRAIN_OUT/TRAINING_DONE" ]]; do
    if [[ -e "$TRAIN_OUT/FAILED" ]]; then
        echo "adapter training failed before gate" >&2
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

mkdir -p "$TRAIN_OUT/checkpoint_gate_models"
ADAPTER_ARGS=()
for epoch in 40 80 120 160 200 235; do
    checkpoint="$TRAIN_OUT/checkpoints/${NAME}_run-${SEED}_epoch-${epoch}.pt"
    model="$TRAIN_OUT/checkpoint_gate_models/epoch${epoch}.model"
    if [[ -s "$checkpoint" ]]; then
        "$CONDA" run -n "$CONDA_ENV" python \
            "$ROOT/scripts/smearing_kink/materialize_mace_checkpoint.py" \
            --template-model "$FINAL_MODEL" --checkpoint "$checkpoint" \
            --output "$model"
        ADAPTER_ARGS+=(--adapter "epoch${epoch}=$model")
    fi
done
ADAPTER_ARGS+=(--adapter "final=$FINAL_MODEL")

"$CONDA" run --no-capture-output -n "$CONDA_ENV" python \
    "$ROOT/scripts/smearing_kink/evaluate_graphene_r2d_additive_adapter.py" \
    --base-model "$BASE_MODEL" --depth3-model "$DEPTH3_MODEL" \
    "${ADAPTER_ARGS[@]}" \
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
echo "R2D additive checkpoint gate complete: $OUTPUT"
