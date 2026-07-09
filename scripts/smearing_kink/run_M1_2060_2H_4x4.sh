#!/usr/bin/env bash
# M1 (2060): as Box B relays each 2H 4×4 Path-P dataset, fine-tune + SSCHA it.
# Compares soft-mode(T_lat) 4×4 vs 3×3 (~0) — does 2H (L)-inert hold at 4×4?
set -uo pipefail
cd ~/phonon
PY=/home/howardwang/miniconda3/envs/phonon/bin/python
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for N in 2H-TaSe2 NbSe2 2H-TaS2; do
  D=data/path_p_${N}_4x4
  echo "[M1-2060] $(date) waiting for $N 4x4 data (.M1_relayed)..."
  while [ ! -f "$D/.M1_relayed" ]; do sleep 120; done
  echo "[M1-2060] $(date) $N 4x4 data arrived -> fine-tune + SSCHA"
  OUT=results/finetune_path_p_${N}_4x4
  W=results/v100/fc2_${N}_4x4_0.015
  Y=$W/${N}_phonopy.yaml
  TAG=${N}_4x4
  if [ ! -f "$OUT/ft.model" ]; then
    echo "[M1-2060] $N fine-tune START $(date +%T)"
    DATA_DIR=$D OUT_DIR=$OUT NAME=ft bash scripts/finetune_pathp_nbse2.sh 150 cuda 2>&1 | tail -8
  else
    echo "[M1-2060] $N ft.model exists, skip fine-tune"
  fi
  echo "[M1-2060] $N SSCHA START $(date +%T)"
  $PY scripts/vq3e_nbse2_sscha.py --phonopy "$Y" --model "$OUT/ft.model" \
    --temperatures 20,100,200,300 --nconfigs 300 --maxpop 5 --device cuda --tag "$TAG" 2>&1 | tail -20
  echo "[M1-2060] $N 4x4 SSCHA DONE $(date)"
  echo "=== $N 4x4 vs 3x3 ==="
  cat results/td_phonon/${TAG}.csv 2>/dev/null
done
touch results/td_phonon/.M1_2060_all_done
echo "[M1-2060] ALL 2H 4x4 DONE $(date)"
