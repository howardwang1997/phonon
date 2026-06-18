#!/usr/bin/env bash
# A3: fine-tune MACE-MP-0 (small) by distilling DFPT force constants.
# Force-dominated (our distilled energies are relative -> energy heavily
# downweighted; forces carry the phonon curvature). Runs in the phonon-mace env.
#
# Usage:  bash scripts/finetune_mace.sh [max_epochs]
set -e
EPOCHS="${1:-30}"
OUT=results/finetune_mace
mkdir -p "$OUT"

# Derive E0s (zeros, since distilled energies are relative) for EXACTLY the
# elements present in the training data — avoids element-count mismatches.
E0S=$(conda run -n phonon-mace python -c "
from ase.io import read
zs = sorted({int(z) for a in read('data/finetune/train.xyz', ':') for z in a.numbers})
print('{' + ','.join(f'{z}:0.0' for z in zs) + '}')
")
echo "E0s = $E0S"

conda run -n phonon-mace mace_run_train \
  --name ft_phonon \
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
  --batch_size 16 \
  --valid_batch_size 32 \
  --eval_interval 1 \
  --lr 0.001 \
  --default_dtype float32 \
  --device cpu \
  --seed 1 \
  --model_dir "$OUT" \
  --results_dir "$OUT/results" \
  --log_dir "$OUT/logs" \
  --checkpoints_dir "$OUT/checkpoints" \
  --save_cpu
