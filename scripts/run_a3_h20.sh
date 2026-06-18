#!/usr/bin/env bash
# A3 PROPER fine-tune on a big GPU (H20, 96 GB): the multihead-replay run the
# 8 GB card couldn't do. Generates the full 16-material distilled dataset
# (no size filter), fine-tunes MACE-MP-0 with an MP-replay head (anti-forgetting)
# + full element coverage on GPU, then evals before/after on train + held-out.
set -e
cd ~/phonon
PY=$HOME/miniconda3/envs/phonon/bin/python
MT=$HOME/miniconda3/envs/phonon/bin/mace_run_train
OUT=results/finetune_h20
EPOCHS="${1:-80}"
rm -rf "$OUT"; mkdir -p "$OUT"

TRAIN="mp-149 mp-1265 mp-22862 mp-661 mp-9946 mp-804 mp-2172 mp-1986 mp-23193 mp-2741 mp-2472 mp-406 mp-4651 mp-1143 mp-2490 mp-1960"
HOLD="mp-7140 mp-984 mp-1138 mp-2605 mp-20351 mp-390 mp-23251 mp-1342 mp-1070 mp-380"

echo "=== 1/3 generate full diverse dataset (no size filter) ==="
$PY scripts/make_finetune_data.py --out data/finetune --train $TRAIN --n-configs 30 --n-single-sites 4

E0S=$($PY -c "from ase.io import read; zs=sorted({int(z) for a in read('data/finetune/train.xyz',':') for z in a.numbers}); print('{'+','.join(f'{z}:0.0' for z in zs)+'}')")
echo "E0s=$E0S"

echo "=== 2/3 multihead-replay fine-tune (GPU) ==="
$MT --name ft_h20 --foundation_model small \
  --multiheads_finetuning True --num_samples_pt 5000 --subselect_pt random \
  --foundation_model_elements True \
  --train_file data/finetune/train.xyz --valid_file data/finetune/val.xyz \
  --energy_key REF_energy --forces_key REF_forces --E0s "$E0S" \
  --energy_weight 0.01 --forces_weight 100.0 --max_num_epochs "$EPOCHS" \
  --batch_size 32 --valid_batch_size 32 --eval_interval 5 --lr 0.001 \
  --default_dtype float32 --device cuda --seed 1 \
  --model_dir "$OUT" --results_dir "$OUT/results" \
  --log_dir "$OUT/logs" --checkpoints_dir "$OUT/checkpoints" --save_cpu

echo "=== 3/3 eval before/after (GPU) ==="
$PY scripts/eval_finetune.py --baseline small --ft-model "$OUT/ft_h20.model" --device cuda \
  --train $TRAIN --holdout $HOLD --out results/finetune_eval_h20.csv
echo "=== A3_H20 COMPLETE ==="
