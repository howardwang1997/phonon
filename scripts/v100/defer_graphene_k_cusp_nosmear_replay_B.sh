#!/usr/bin/env bash
# Delay the GPU QE replay on storage-constrained V100-B until the CPU pilot is bundled.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
WORK_ROOT="${WORK_ROOT:-/data/graphene_k_cusp_nosmear}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MIN_FREE_GIB="${MIN_FREE_GIB:-55}"
SOURCE_LANE="$WORK_ROOT/lanes/pilot_B_k192_qe75_conda"
SOURCE_RESPONSE="$WORK_ROOT/sets/k192_tetra_qe75_conda/tmp/_ph0"
REPLAY_SET="$WORK_ROOT/sets/k192_tetra_qe75_npk120k"
REPLAY_LANE="$WORK_ROOT/lanes/pilot_B_k192_qe75_npk120k"
RUNNER="$ROOT/scripts/v100/run_graphene_k_cusp_nosmear_dfpt_npk.sh"
LOG="$REPLAY_LANE/deferred.log"

mkdir -p "$REPLAY_LANE"
exec 9>"$REPLAY_LANE/.deferred.lock"
if ! flock -n 9; then
    echo "[defer-replay-B] another deferred launcher is active"
    exit 0
fi
exec > >(tee -a "$LOG") 2>&1

echo "[defer-replay-B] wait_start=$(date -Is)"
while [[ ! -s "$SOURCE_LANE/DONE" ]]; do
    if [[ -s "$SOURCE_LANE/FAILED" ]]; then
        echo "[defer-replay-B] source pilot failed; replay remains stopped" >&2
        exit 2
    fi
    sleep "$POLL_SECONDS"
done

# DONE is written only after the immutable bundle and manifest are complete.
test -s "$SOURCE_LANE/manifest.json"
test -s "$SOURCE_LANE/bundle/K/gr.dyn"
test -s "$SOURCE_LANE/bundle/KM_d007/gr.dyn"
grep -q '"status": "complete"' "$SOURCE_LANE/manifest.json"
echo "[defer-replay-B] source_bundle_verified=$(date -Is)"

# This exact directory contains ph.x response wavefunctions; all persistent
# outputs have already been copied into the source lane bundle above.
if [[ -d "$SOURCE_RESPONSE" ]]; then
    case "$SOURCE_RESPONSE" in
        /data/graphene_k_cusp_nosmear/sets/k192_tetra_qe75_conda/tmp/_ph0) ;;
        *) echo "refuse unexpected cleanup target: $SOURCE_RESPONSE" >&2; exit 8 ;;
    esac
    du -sh "$SOURCE_RESPONSE"
    rm -rf -- "$SOURCE_RESPONSE"
    echo "[defer-replay-B] removed_reproducible_response_scratch=$(date -Is)"
fi

free_kib="$(df -Pk "$WORK_ROOT" | awk 'NR==2 {print $4}')"
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
if (( free_kib < required_kib )); then
    echo "[defer-replay-B] storage gate failed after cleanup: free_kib=$free_kib" >&2
    exit 5
fi

# Remove only an incomplete scratch set left by the intentionally stopped
# replay.  Its lane log is retained as provenance.
if [[ -e "$REPLAY_SET" && ! -s "$REPLAY_LANE/DONE" ]]; then
    case "$REPLAY_SET" in
        /data/graphene_k_cusp_nosmear/sets/k192_tetra_qe75_npk120k) ;;
        *) echo "refuse unexpected replay target: $REPLAY_SET" >&2; exit 8 ;;
    esac
    rm -rf -- "$REPLAY_SET"
fi

echo "[defer-replay-B] replay_start=$(date -Is)"
exec env \
    ROOT="$ROOT" \
    CONDA=/root/miniconda3/bin/conda \
    QEBIN=/data/qe-7.5-npk120k/bin \
    QE_ENV_SCRIPT=/root/nvhpc_env.sh \
    KGRID=192 \
    QE_TAG=qe75_npk120k \
    NP=1 \
    OMP_THREADS=4 \
    MIN_FREE_GIB="$MIN_FREE_GIB" \
    CUDA_VISIBLE_DEVICES=0 \
    "$RUNNER" B pilot
