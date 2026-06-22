#!/usr/bin/env bash
# Fixed 3rd-order distillation: mixed-amplitude anharmonic data (preserves
# harmonic) + REPLAY (pt1000, anti-forgetting) fine-tune from the harmonic-FT
# model -> keep κ=113 gain AND add the cubic correction. Re-eval Si κ.
set -uo pipefail
cd ~/phonon
PY=~/miniconda3/envs/phonon/bin/python
MT=~/miniconda3/envs/phonon/bin/mace_run_train
MAT=Si
BASE=results/ablation/br_B16_s1/ft.model

$PY scripts/make_anharm_data.py --material $MAT --fc-dir results/fc3 --out data/anharm2_$MAT \
  --n-configs 300 --rattle-std 0.06
E0S=$($PY -c "from ase.io import read; zs=sorted({int(z) for a in read('data/anharm2_${MAT}/train.xyz',':') for z in a.numbers}); print('{'+','.join(f'{z}:0.0' for z in zs)+'}')")
OUT=results/ablation/anharm2_$MAT; rm -rf "$OUT"; mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 $MT --name ft --foundation_model "$BASE" \
  --multiheads_finetuning True --num_samples_pt 1000 --subselect_pt random \
  --foundation_model_elements True \
  --train_file data/anharm2_${MAT}/train.xyz --valid_file data/anharm2_${MAT}/val.xyz \
  --energy_key REF_energy --forces_key REF_forces --E0s "$E0S" \
  --energy_weight 0.1 --forces_weight 100.0 --max_num_epochs 60 \
  --batch_size 16 --valid_batch_size 16 --eval_interval 20 --lr 0.0005 \
  --default_dtype float32 --device cuda --seed 1 \
  --model_dir "$OUT" --results_dir "$OUT" --log_dir "$OUT" --checkpoints_dir "$OUT" --save_cpu

CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mlip_kappa.py --material $MAT --model "$OUT/ft.model" \
  --sc 3 --mesh 21 --device cuda --out results/kappa/Si_anharm2.json
echo "[anharm2] DONE -> results/kappa/Si_anharm2.json"
