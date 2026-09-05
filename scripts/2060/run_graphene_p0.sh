#!/usr/bin/env bash
# Restartable supervisor for the two 2060 P0 deployments.  Each Python job
# caches every completed fc2, and this wrapper retries transient failures.
set -uo pipefail

ROOT="${ROOT:-$HOME/phonon}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
OUT="$ROOT/results/p0_graphene"
mkdir -p "$OUT"
cd "$ROOT"

run_one() {
  local dataset="$1"
  local log="$OUT/${dataset}_run.log"
  local attempt=1
  while (( attempt <= 3 )); do
    echo "=== P0 $dataset attempt=$attempt start $(date -Is) ===" | tee -a "$log"
    if "$CONDA" run --no-capture-output -n phonon python \
      scripts/smearing_kink/graphene_p0_deploy.py \
      --dataset "$dataset" --device cuda 2>&1 | tee -a "$log"; then
      touch "$OUT/${dataset}.DONE"
      rm -f "$OUT/${dataset}.FAILED"
      echo "=== P0 $dataset COMPLETE $(date -Is) ===" | tee -a "$log"
      return 0
    fi
    echo "=== P0 $dataset failed attempt=$attempt $(date -Is) ===" | tee -a "$log"
    attempt=$((attempt + 1))
  done
  touch "$OUT/${dataset}.FAILED"
  return 1
}

status=0
run_one 8x8 || status=1
run_one 15pt || status=1
exit "$status"
