#!/usr/bin/env bash
# 2060 (L)-channel: process any material whose Path-P data is ready (readiness-order, not fixed).
# NbSe2 reused. Idempotent: skips a material whose fine-tune model already exists.
set -u
PY=/home/howardwang/miniconda3/envs/phonon/bin/python
export LD_LIBRARY_PATH="$(cd $(dirname $PY)/.. && pwd)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/howardwang/phonon
FT=scripts/finetune_pathp_nbse2.sh
SSCHA=scripts/vq3e_nbse2_sscha.py
declare -A FC2=( [1T-VSe2]=results/v100/fc2_1T-VSe2_4_4_1_0.015/1T-VSe2_phonopy.yaml [NbS2]=results/v100/fc2_nbs2_3x3_0.015/NbS2_phonopy.yaml [2H-TaS2]=results/v100/fc2_2H-TaS2_3x3_0.015/2H-TaS2_phonopy.yaml [2H-TaSe2]=results/v100/fc2_tase2_3x3_0.015/2H-TaSe2_phonopy.yaml [1T-TiSe2]=results/v100/fc2_tise2_4x4_0.015/1T-TiSe2_phonopy.yaml )
MATS="1T-VSe2 NbS2 2H-TaS2 2H-TaSe2 1T-TiSe2"
done_all() { for M in $MATS; do [ -f results/finetune_path_p_${M}/ft.model ] || return 1; done; return 0; }
while ! done_all; do
  for M in $MATS; do
    OUT=results/finetune_path_p_${M}; D=data/path_p_${M}
    [ -f "$OUT/ft.model" ] && continue                    # already fine-tuned
    [ -f "$D/train.xyz" ] || continue                      # data not ready yet
    echo "[2060-ft] $M $(date +%T)"
    DATA_DIR=$D OUT_DIR=$OUT NAME=ft bash $FT 150 cuda 2>&1 | tail -3
    [ -f "$OUT/ft.model" ] || { echo "[2060-ft] $M FAILED"; continue; }
    echo "[2060-sscha] $M $(date +%T)"
    $PY $SSCHA --phonopy "${FC2[$M]}" --model "$OUT/ft.model" --tag ${M}_L 2>&1 | grep -E "sscha|crossover|min_freq|Error" | tail -5
    echo "[2060] $M DONE $(date)"
  done
  sleep 60
done
echo "[2060] ALL L-DONE $(date)"
