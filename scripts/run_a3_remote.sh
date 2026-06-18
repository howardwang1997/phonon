#!/usr/bin/env bash
# A3 on the remote host: train on CPU (8 GB VRAM can't hold full-element MACE
# force double-backprop, fixed ~7 GiB), then eval inference on the GPU (fast,
# fits). Diverse 10-material distilled training set (<=80-atom cells).
set -e
cd ~/phonon
PY=$HOME/miniconda3/envs/phonon/bin/python
MT=$HOME/miniconda3/envs/phonon/bin/mace_run_train
OUT=results/finetune_mace_gpu
EPOCHS="${1:-30}"
rm -rf "$OUT"; mkdir -p "$OUT"

E0S=$($PY -c "from ase.io import read; zs=sorted({int(z) for a in read('data/finetune/train.xyz',':') for z in a.numbers}); print('{'+','.join(f'{z}:0.0' for z in zs)+'}')")
echo "E0s=$E0S"

echo "=== TRAIN (CPU, 16 cores) ==="
$MT --name ft_phonon_gpu --foundation_model small \
  --multiheads_finetuning False --foundation_model_elements True \
  --train_file data/finetune/train.xyz --valid_file data/finetune/val.xyz \
  --energy_key REF_energy --forces_key REF_forces --E0s "$E0S" \
  --energy_weight 0.01 --forces_weight 100.0 --max_num_epochs "$EPOCHS" \
  --batch_size 8 --valid_batch_size 8 --eval_interval 2 --lr 0.001 \
  --default_dtype float32 --device cpu --seed 1 \
  --model_dir "$OUT" --results_dir "$OUT/results" \
  --log_dir "$OUT/logs" --checkpoints_dir "$OUT/checkpoints" --save_cpu

echo "=== TRAIN DONE; EVAL on GPU (inference) ==="
$PY scripts/eval_finetune.py --baseline small \
  --ft-model "$OUT/ft_phonon_gpu.model" --device cuda \
  --train mp-149 mp-1265 mp-22862 mp-2172 mp-1986 mp-23193 mp-2472 mp-406 mp-2490 mp-4651 \
  --holdout mp-661 mp-804 mp-9946 mp-7140 mp-984 mp-1138 mp-2605 mp-20351 mp-390 \
  --out results/finetune_eval_gpu.csv
echo "=== A3_REMOTE COMPLETE ==="
