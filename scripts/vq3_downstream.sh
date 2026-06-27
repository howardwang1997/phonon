#!/bin/bash
# V-Q3 downstream: distill the NbSe2 DFT fc2 -> harmonic data -> fine-tune MACE.
# The decisive Gate#2 test follows (eval foundation vs FT dispersion): does the
# distilled MLIP recover the CDW soft mode that all foundation backbones miss?
set -uo pipefail
cd /root/phonon
export LD_LIBRARY_PATH=/root/miniconda3/envs/phonon/lib
PY=/root/miniconda3/envs/phonon/bin/python
echo "=== distill NbSe2 DFT fc2 -> data $(date) ==="
$PY scripts/m1_1b_make_graphene_data.py \
  --phonopy results/vq3/nbse2_dft_phonopy.yaml \
  --out data/finetune_nbse2 --n-configs 60 --rattle-std 0.03 || exit 2
echo "=== fine-tune MACE on NbSe2 distillation $(date) ==="
DATA_DIR=data/finetune_nbse2 OUT_DIR=results/finetune_nbse2 NAME=ft_nbse2 \
  bash scripts/finetune_graphene.sh 80 cpu || exit 3
echo "=== DOWNSTREAM DONE $(date) ==="
