#!/usr/bin/env bash
# Dispatch ONE campaign job into the right conda env and run it.
# Called by queue_worker.sh with the GPU already pinned via CUDA_VISIBLE_DEVICES.
# The command string is passed via the $CMD env (avoids TSV/quoting issues).
#   args: ENVNAME GPUFLAG GROUP DONE_MARKER JOB_ID
set -uo pipefail
cd "$HOME/phonon"
ENVNAME="$1"; GPUFLAG="$2"; GROUP="$3"; DONE="$4"; JID="$5"
CONDA="$HOME/miniconda3"

# logical env -> conda env name (mace/mace-omat share the phonon env)
case "$ENVNAME" in
  phonon|mace|mace-omat) CE=phonon ;;
  *)                     CE="$ENVNAME" ;;
esac

source "$CONDA/etc/profile.d/conda.sh"
if ! conda activate "$CE" 2>/dev/null; then
  mkdir -p results/h20/joblogs
  echo "[$JID] conda env '$CE' missing (bootstrap incomplete) -> skip" \
    | tee "results/h20/joblogs/$JID.log"
  exit 9
fi
# per-env runtime libs (libstdc++/lapack live in the env; see bootstrap gotchas)
export LD_LIBRARY_PATH="$CONDA/envs/$CE/lib:${LD_LIBRARY_PATH:-}"

# thread caps: phono3py RTA (e9) is CPU-bound -> more threads; everything else 4
if [ "$GROUP" = "e9" ]; then
  export OMP_NUM_THREADS="${OMP_THREADS_KAPPA:-16}"
else
  export OMP_NUM_THREADS="${OMP_THREADS:-4}"
fi
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

mkdir -p results/h20/joblogs
LOG="results/h20/joblogs/$JID.log"
echo "=== [$JID] gpu=${CUDA_VISIBLE_DEVICES:-?} env=$CE group=$GROUP $(date) ===" | tee "$LOG"

bash -c "$CMD" >>"$LOG" 2>&1
rc=$?

if [ -f "$DONE" ]; then
  echo "=== [$JID] DONE rc=$rc $(date) ===" | tee -a "$LOG"
  exit 0
else
  echo "=== [$JID] FAIL rc=$rc (no done-marker: $DONE) $(date) ===" | tee -a "$LOG"
  exit 1
fi
