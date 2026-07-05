#!/usr/bin/env bash
# 2060 (L)-channel: per material, wait for Path-P train.xyz (rsynced from boxes) -> fine-tune -> SSCHA.
# NbSe2 reused (skips if model exists).
set -u
PY=/home/howardwang/miniconda3/envs/phonon/bin/python
export LD_LIBRARY_PATH="$(cd $(dirname $PY)/.. && pwd)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CD=/home/howardwang/phonon; cd $CD
FT=scripts/finetune_pathp_nbse2.sh
SSCHA=scripts/vq3e_nbse2_sscha.py
# material : fc2 yaml (for SSCHA --phonopy)
declare -A FC2=( [1T-VSe2]=results/v100/fc2_1T-VSe2_4_4_1_0.015/1T-VSe2_phonopy.yaml [NbS2]=results/v100/fc2_nbs2_3x3_0.015/NbS2_phonopy.yaml [2H-TaS2]=results/v100/fc2_2H-TaS2_3x3_0.015/2H-TaS2_phonopy.yaml [2H-TaSe2]=results/v100/fc2_tase2_3x3_0.015/2H-TaSe2_phonopy.yaml [1T-TiSe2]=results/v100/fc2_tise2_4x4_0.015/1T-TiSe2_phonopy.yaml [NbSe2]=results/vq3/nbse2_dft_phonopy.yaml )
for M in 1T-VSe2 NbS2 2H-TaS2 2H-TaSe2 1T-TiSe2 NbSe2; do
  D=data/path_p_${M}; OUT=results/finetune_path_p_${M}
  # NbSe2 reuse: skip if Path-P model already exists from prior Stage C
  if [ "$M" = "NbSe2" ] && [ -f results/finetune_path_p_nbse2/ft.model ]; then echo "[NbSe2] reuse existing Path-P model"; continue; fi
  echo "[2060] waiting for $M Path-P data..."
  until [ -f "$D/train.xyz" ]; do sleep 120; done
  echo "[2060-ft] $M $(date +%T)"
  DATA_DIR=$D OUT_DIR=$OUT NAME=ft bash $FT 150 cuda 2>&1 | tail -3
  MODEL=$OUT/ft.model
  [ -f "$MODEL" ] || { echo "[2060-ft] $M fine-tune FAILED"; continue; }
  echo "[2060-sscha] $M $(date +%T)"
  $PY $SSCHA --phonopy "${FC2[$M]}" --model "$MODEL" --tag ${M}_L 2>&1 | grep -E "sscha|crossover|min_freq|Error" | tail -5
  echo "[2060] $M DONE $(date)"
done
echo "[2060] ALL L-DONE $(date)"
