#!/usr/bin/env bash
# A3 ablation: sweep replay strength (num_samples_pt) to find the
# robustness-vs-accuracy sweet spot.
#   pt=0    -> single-head (no replay): best accuracy, catastrophic on unseen elems
#   pt=5000 -> full multihead: robust (unseen-elem imag fixed), lower accuracy
# Goal: smallest replay that keeps SiC/BN imaginary modes ~0 without tanking
# in-domain MAE or held-out (all-seen) transfer.
#
# Reuses the cached replay file (~/.cache/mace/mp_traj_combinedxyz) and cached
# MDR data. Baseline numbers come from the prior run (eval is --ft-only).
set -e
cd ~/phonon
PY=$HOME/miniconda3/envs/phonon/bin/python
MT=$HOME/miniconda3/envs/phonon/bin/mace_run_train
mkdir -p results/ablation

TRAIN="mp-149 mp-1265 mp-22862 mp-661 mp-9946 mp-804 mp-2172 mp-1986 mp-23193 mp-2741 mp-2472 mp-406 mp-4651 mp-1143 mp-2490 mp-1960"
HOLD="mp-7140 mp-984 mp-1138 mp-2605 mp-20351 mp-390"
EPOCHS="${1:-50}"
PTS="${2:-0 1000 2500 5000}"

echo "=== generate dataset (once) ==="
$PY scripts/make_finetune_data.py --out data/finetune --train $TRAIN --n-configs 30 --n-single-sites 4
E0S=$($PY -c "from ase.io import read; zs=sorted({int(z) for a in read('data/finetune/train.xyz',':') for z in a.numbers}); print('{'+','.join(f'{z}:0.0' for z in zs)+'}')")
echo "E0s=$E0S"

for PT in $PTS; do
  echo "=== TRAIN pt=$PT (epochs=$EPOCHS) ==="
  OUT=results/ablation/pt$PT; rm -rf "$OUT"; mkdir -p "$OUT"
  if [ "$PT" = "0" ]; then
    MH="--multiheads_finetuning False"
  else
    MH="--multiheads_finetuning True --num_samples_pt $PT --subselect_pt random"
  fi
  $MT --name ft --foundation_model small $MH --foundation_model_elements True \
    --train_file data/finetune/train.xyz --valid_file data/finetune/val.xyz \
    --energy_key REF_energy --forces_key REF_forces --E0s "$E0S" \
    --energy_weight 0.01 --forces_weight 100.0 --max_num_epochs "$EPOCHS" \
    --batch_size 32 --valid_batch_size 32 --eval_interval 10 --lr 0.001 \
    --default_dtype float32 --device cuda --seed 1 \
    --model_dir "$OUT" --results_dir "$OUT" --log_dir "$OUT" --checkpoints_dir "$OUT" --save_cpu
  echo "=== EVAL pt=$PT ==="
  $PY scripts/eval_finetune.py --baseline small --ft-model "$OUT/ft.model" --device cuda --ft-only \
    --train $TRAIN --holdout $HOLD --out "results/ablation/eval_pt$PT.csv"
  echo "=== pt=$PT DONE ==="
done
echo "=== ABLATION COMPLETE ==="
