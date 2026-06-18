#!/usr/bin/env bash
# A3 third anti-forgetting comparison: LoRA fine-tuning.
# Freeze the foundation model, train only low-rank adapters -> the base's
# knowledge (incl. unseen elements C/B/In) is largely preserved by construction,
# so this should curb the catastrophic forgetting of full single-head FT
# WITHOUT needing any replay data. Sweep lora_rank for the capacity/drift knob.
#
# Compare against: pt=0 single-head full FT (results/ablation/eval_pt0.csv) and
# multihead replay (results/ablation/eval_pt5000.csv). Same data/epochs/batch.
set -e
cd ~/phonon
PY=$HOME/miniconda3/envs/phonon/bin/python
MT=$HOME/miniconda3/envs/phonon/bin/mace_run_train
mkdir -p results/ablation

TRAIN="mp-149 mp-1265 mp-22862 mp-661 mp-9946 mp-804 mp-2172 mp-1986 mp-23193 mp-2741 mp-2472 mp-406 mp-4651 mp-1143 mp-2490 mp-1960"
HOLD="mp-7140 mp-984 mp-1138 mp-2605 mp-20351 mp-390"
EPOCHS="${1:-50}"
RANKS="${2:-8 32}"   # lora_rank values (alpha set = rank, i.e. scale ~1)

# dataset (idempotent; uses cached MDR)
$PY scripts/make_finetune_data.py --out data/finetune --train $TRAIN --n-configs 30 --n-single-sites 4
E0S=$($PY -c "from ase.io import read; zs=sorted({int(z) for a in read('data/finetune/train.xyz',':') for z in a.numbers}); print('{'+','.join(f'{z}:0.0' for z in zs)+'}')")
echo "E0s=$E0S"

for R in $RANKS; do
  echo "=== TRAIN lora_rank=$R (alpha=$R, epochs=$EPOCHS) ==="
  OUT=results/ablation/lora$R; rm -rf "$OUT"; mkdir -p "$OUT"
  $MT --name ft --foundation_model small \
    --multiheads_finetuning False --foundation_model_elements True \
    --lora True --lora_rank "$R" --lora_alpha "$R" \
    --train_file data/finetune/train.xyz --valid_file data/finetune/val.xyz \
    --energy_key REF_energy --forces_key REF_forces --E0s "$E0S" \
    --energy_weight 0.01 --forces_weight 100.0 --max_num_epochs "$EPOCHS" \
    --batch_size 32 --valid_batch_size 32 --eval_interval 10 --lr 0.001 \
    --default_dtype float32 --device cuda --seed 1 \
    --model_dir "$OUT" --results_dir "$OUT" --log_dir "$OUT" --checkpoints_dir "$OUT" --save_cpu
  echo "=== EVAL lora_rank=$R ==="
  $PY scripts/eval_finetune.py --baseline small --ft-model "$OUT/ft.model" --device cuda --ft-only \
    --train $TRAIN --holdout $HOLD --out "results/ablation/eval_lora$R.csv"
  echo "=== lora_rank=$R DONE ==="
done
echo "=== LORA SWEEP COMPLETE ==="
