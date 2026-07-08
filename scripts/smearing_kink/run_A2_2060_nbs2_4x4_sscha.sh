#!/usr/bin/env bash
# A2 (2060 side): NbS2 4x4 fine-tune + SSCHA — 2H honesty convergence test.
# 4x4 Path-P data (data/path_p_NbS2_4x4) + 4x4 fc2 already on 2060.
# Compare soft-mode(T_lat) to the 3x3 result (NbS2_L.csv: ~0 all T).
# If 4x4 also ~0 -> 2H (L)-inert is NOT a 3x3 artifact, honesty caveat CLOSED.
# NbS2 = deepest bare soft mode (-103 cm-1) among 2H -> the strongest test.
set -uo pipefail
cd ~/phonon
PY=/home/howardwang/miniconda3/envs/phonon/bin/python
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=data/path_p_NbS2_4x4
OUT=results/finetune_path_p_NbS2_4x4
FC2=results/v100/fc2_nbs2_4x4_0.015/NbS2_phonopy.yaml
TAG=NbS2_4x4

echo "[A2-4x4] START $(date)"
# 1. fine-tune (skip if done)
if [ ! -f "$OUT/ft.model" ]; then
  echo "[A2-4x4] fine-tune START $(date +%T)"
  DATA_DIR=$DATA OUT_DIR=$OUT NAME=ft bash scripts/finetune_pathp_nbse2.sh 150 cuda 2>&1 | tail -10
else
  echo "[A2-4x4] ft.model exists, skip fine-tune"
fi
# 2. SSCHA
echo "[A2-4x4] SSCHA START $(date +%T)"
$PY scripts/vq3e_nbse2_sscha.py --phonopy "$FC2" --model "$OUT/ft.model" \
  --temperatures 20,100,200,300 --nconfigs 300 --maxpop 5 --device cuda --tag "$TAG" 2>&1 | tail -25
echo "[A2-4x4] DONE $(date)"
echo "=== NbS2 4x4 (L) vs 3x3 ==="
cat results/td_phonon/${TAG}.csv 2>/dev/null
echo "--- 3x3 (NbS2_L.csv) ---"; cat results/td_phonon/NbS2_L.csv
touch results/td_phonon/.A2_4x4_done
