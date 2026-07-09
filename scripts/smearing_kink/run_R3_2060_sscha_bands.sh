#!/usr/bin/env bash
# R3: SSCHA free-energy Hessian -> phonon band structure at each T_lat (Gamma-M-K-Gamma)
# for VSe2 (1T) + NbSe2 (2H). Gives the (L)-temperature phonon SPECTRA. ~30min/material.
set -uo pipefail
cd ~/phonon
PY=/home/howardwang/miniconda3/envs/phonon/bin/python
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[R3] VSe2 SSCHA-bands START $(date)"
$PY scripts/smearing_kink/vq3e_sscha_bands.py \
  --phonopy results/vq_family/vse2/1T-VSe2_dg0.020.yaml \
  --model results/finetune_path_p_1T-VSe2/ft.model \
  --tag VSe2_Lband --device cuda 2>&1 | tail -12
echo "[R3] NbSe2 SSCHA-bands START $(date)"
$PY scripts/smearing_kink/vq3e_sscha_bands.py \
  --phonopy results/v100/fc2_nbse2_3x3_0.020/NbSe2_phonopy.yaml \
  --model results/finetune_path_p_nbse2/ft.model \
  --tag NbSe2_Lband --device cuda 2>&1 | tail -12
touch results/td_phonon/.R3_done
echo "[R3] DONE $(date)"
