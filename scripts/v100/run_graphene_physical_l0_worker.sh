#!/usr/bin/env bash
# Continue one frozen graphene L0 trajectory on an isolated V100 deployment.
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "usage: $0 TEMPERATURE_K SEED [formal|smoke]" >&2
    exit 2
fi

TEMPERATURE="$1"
SEED="$2"
MODE="${3:-formal}"
ROOT="${DIST_ROOT:-/data/graphene_physical_s0_l0_dist/project}"
PHONON_ENV="${PHONON_ENV:-phonon}"
if [[ -n "${CONDA:-}" ]]; then
    :
elif [[ -x /root/miniconda3/bin/conda ]]; then
    CONDA=/root/miniconda3/bin/conda
elif [[ -x /home/howardwang/miniconda3/bin/conda ]]; then
    CONDA=/home/howardwang/miniconda3/bin/conda
else
    echo "cannot locate Conda; set CONDA to an executable path" >&2
    exit 2
fi
CONDA_BASE="$($CONDA info --base)"
S0="$ROOT/results/graphene_physics_temperature/post_p4_feasibility/S0_unified_short"
FORMAL="$S0/formal_240ep_2060_seed83_replayw16"
OUT="$S0/L0_classical_tdep"
DATA="$ROOT/data/graphene_physical_s0"
BASE="$ROOT/results/gr_backbone_v11/ft_graphene.model"
DELTA="$FORMAL/selected_checkpoint.model"
BG="$ROOT/results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
VERIFY="$ROOT/scripts/smearing_kink/verify_graphene_physical_l0_worker.py"

case "$TEMPERATURE" in
    300|600) ;;
    *) echo "worker accepts only frozen development temperatures 300 or 600 K" >&2; exit 2 ;;
esac
case "$SEED" in
    0|2) DT=0.5; EQUIL=3000; STRIDE=80; DT_TAG=dt0p5 ;;
    1) DT=0.25; EQUIL=6000; STRIDE=160; DT_TAG=dt0p25 ;;
    *) echo "worker seed must be 0, 1, or 2" >&2; exit 2 ;;
esac
case "$MODE" in
    formal)
        TARGET=3000
        LANE="$OUT/T${TEMPERATURE}"
        CHECKPOINT_ROOT="$LANE/checkpoints/seed${SEED}_${DT_TAG}"
        CHECKPOINT="$CHECKPOINT_ROOT/T${TEMPERATURE}"
        SHORT_TDEP="$LANE/final_short_seed${SEED}.npz"
        RESULT="$LANE/worker_seed${SEED}_result.json"
        ALLOW="$ROOT/coordination/ALLOW_T${TEMPERATURE}"
        if [[ ! -s "$ALLOW" ]]; then
            echo "refusing formal worker: missing gate token $ALLOW" >&2
            exit 3
        fi
        ;;
    smoke)
        TARGET="${SMOKE_TARGET:-121}"
        SOURCE="$OUT/T${TEMPERATURE}/checkpoints/seed${SEED}_${DT_TAG}/T${TEMPERATURE}"
        LANE="$ROOT/smoke/T${TEMPERATURE}_seed${SEED}"
        CHECKPOINT_ROOT="$LANE/checkpoints/seed${SEED}_${DT_TAG}"
        CHECKPOINT="$CHECKPOINT_ROOT/T${TEMPERATURE}"
        SHORT_TDEP="$LANE/smoke_short_seed${SEED}.npz"
        RESULT="$LANE/smoke_result.json"
        if [[ ! -d "$CHECKPOINT" ]]; then
            mkdir -p "$CHECKPOINT_ROOT"
            cp -a "$SOURCE" "$CHECKPOINT_ROOT/"
        fi
        ;;
    *) echo "mode must be formal or smoke" >&2; exit 2 ;;
esac

mkdir -p "$LANE"
LOG="$LANE/worker_seed${SEED}_${MODE}.log"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="$CONDA_BASE/envs/${PHONON_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== isolated V100 L0 worker T=$TEMPERATURE seed=$SEED mode=$MODE START $(date -Is) ==="
"$CONDA" run --no-capture-output -n "$PHONON_ENV" python "$VERIFY" \
    --root "$ROOT" --temperature "$TEMPERATURE" --seed "$SEED" \
    --checkpoint "$CHECKPOINT" --output "$LANE/preflight_seed${SEED}.json"

CURRENT="$($CONDA run -n "$PHONON_ENV" python -c \
    'import numpy as np,sys; print(len(np.load(sys.argv[1],allow_pickle=False)["positions"]))' \
    "$CHECKPOINT/snapshots.npz" | tail -1)"
if (( CURRENT < TARGET )); then
    "$CONDA" run --no-capture-output -n "$PHONON_ENV" python \
        "$ROOT/scripts/smearing_kink/td_phonon_friedel.py" \
        --model "$BASE" --delta-model "$DELTA" --bg "$BG" \
        --operator "$DATA/operators/T${TEMPERATURE}_operator.npz" \
        --friedel-tel "$TEMPERATURE" --device cuda \
        --tag "graphene_physical_s0_L0_T${TEMPERATURE}_seed${SEED}_${MODE}_v100" \
        --temperatures "$TEMPERATURE" --dt "$DT" --equil "$EQUIL" \
        --nsnap "$TARGET" --stride "$STRIDE" --npoints 201 --seed "$SEED" \
        --checkpoint-root "$CHECKPOINT_ROOT" --checkpoint-every 25 \
        --max-temperature-factor 5 --min-pair-distance 0.8 --max-force 100
fi

"$CONDA" run --no-capture-output -n "$PHONON_ENV" python \
    "$ROOT/scripts/smearing_kink/recompute_graphene_tdep_from_checkpoints.py" \
    --background "$BG" --checkpoint "$CHECKPOINT" \
    --subtract-operator "$DATA/operators/T${TEMPERATURE}_operator.npz" \
    --temperature "$TEMPERATURE" \
    --tag "graphene_physical_s0_L0_T${TEMPERATURE}_${MODE}_short_seed${SEED}_v100" \
    --output "$SHORT_TDEP"

"$CONDA" run --no-capture-output -n "$PHONON_ENV" python "$VERIFY" \
    --root "$ROOT" --temperature "$TEMPERATURE" --seed "$SEED" \
    --checkpoint "$CHECKPOINT" --expected-count "$TARGET" \
    --short-tdep "$SHORT_TDEP" --output "$RESULT"
touch "$LANE/WORKER_SEED${SEED}_${MODE^^}_DONE"
echo "=== isolated V100 L0 worker T=$TEMPERATURE seed=$SEED mode=$MODE COMPLETE $(date -Is) ==="
