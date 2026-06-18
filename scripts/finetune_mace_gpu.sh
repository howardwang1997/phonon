#!/usr/bin/env bash
# A3 (proper): multihead-replay fine-tune of MACE-MP-0 on GPU.
# Fixes the catastrophic forgetting seen in the CPU PoC:
#   - --multiheads_finetuning True : a replay head on foundation MP data keeps
#     the model anchored to the broad distribution (and naturally separates the
#     relative-energy phonon head from the absolute-energy MP head)
#   - --foundation_model_elements True : keep the full periodic table
#   - diverse + larger training set (passed via data/finetune)
#
# Usage:  bash scripts/finetune_mace_gpu.sh [max_epochs]
set -e
EPOCHS="${1:-60}"
# 8 GB card: reduce fragmentation (1.4 GB was reserved-but-unallocated at OOM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$HOME/miniconda3/envs/phonon/bin/python"
MACE_TRAIN="$HOME/miniconda3/envs/phonon/bin/mace_run_train"
OUT=results/finetune_mace_gpu
mkdir -p "$OUT"

E0S=$($PY -c "
from ase.io import read
zs = sorted({int(z) for a in read('data/finetune/train.xyz', ':') for z in a.numbers})
print('{' + ','.join(f'{z}:0.0' for z in zs) + '}')
")
echo "E0s = $E0S"

# Single-head fine-tune on diverse distilled data (no MP replay → small cells
# only → fits the 8 GB card). multiheads_finetuning False avoids the OOM from
# large Materials-Project replay structures under force double-backprop.
$MACE_TRAIN \
  --name ft_phonon_gpu \
  --foundation_model small \
  --multiheads_finetuning False \
  --foundation_model_elements True \
  --train_file data/finetune/train.xyz \
  --valid_file data/finetune/val.xyz \
  --energy_key REF_energy \
  --forces_key REF_forces \
  --E0s "$E0S" \
  --energy_weight 0.01 \
  --forces_weight 100.0 \
  --max_num_epochs "$EPOCHS" \
  --batch_size 1 \
  --valid_batch_size 2 \
  --eval_interval 2 \
  --lr 0.001 \
  --default_dtype float32 \
  --device cuda \
  --seed 1 \
  --model_dir "$OUT" \
  --results_dir "$OUT/results" \
  --log_dir "$OUT/logs" \
  --checkpoints_dir "$OUT/checkpoints" \
  --save_cpu
