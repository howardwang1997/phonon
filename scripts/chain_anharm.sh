#!/usr/bin/env bash
# Task 3 downstream: when DFT fc3 is ready -> anharmonic distillation data ->
# fine-tune the harmonic-FT model on it -> re-eval Si kappa (close the residual).
set -uo pipefail
cd ~/phonon
PY=~/miniconda3/envs/phonon/bin/python
MT=~/miniconda3/envs/phonon/bin/mace_run_train
MAT=${MAT:-Si}
BASE=${BASE:-results/ablation/br_B16_s1/ft.model}   # harmonic-FT model to extend

echo "[anharm] waiting for results/fc3/${MAT}_fc3.hdf5 ..."
while [ ! -f "results/fc3/${MAT}_fc3.hdf5" ]; do sleep 60; done
sleep 10

# 1. anharmonic distillation data (cubic Taylor labels, larger rattles)
$PY scripts/make_anharm_data.py --material $MAT --fc-dir results/fc3 --out data/anharm_$MAT \
  --n-configs 250 --rattle-std 0.06

# 2. fine-tune the harmonic-FT model FURTHER on anharmonic data
E0S=$($PY -c "from ase.io import read; zs=sorted({int(z) for a in read('data/anharm_${MAT}/train.xyz',':') for z in a.numbers}); print('{'+','.join(f'{z}:0.0' for z in zs)+'}')")
OUT=results/ablation/anharm_$MAT; rm -rf "$OUT"; mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 $MT --name ft --foundation_model "$BASE" \
  --multiheads_finetuning False --foundation_model_elements True \
  --train_file data/anharm_${MAT}/train.xyz --valid_file data/anharm_${MAT}/val.xyz \
  --energy_key REF_energy --forces_key REF_forces --E0s "$E0S" \
  --energy_weight 0.1 --forces_weight 100.0 --max_num_epochs 80 \
  --batch_size 16 --valid_batch_size 16 --eval_interval 20 --lr 0.001 \
  --default_dtype float32 --device cuda --seed 1 \
  --model_dir "$OUT" --results_dir "$OUT" --log_dir "$OUT" --checkpoints_dir "$OUT" --save_cpu

# 3. re-eval kappa with the anharmonic-distilled model (vs harmonic-FT 113, DFT ~140)
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mlip_kappa.py --material $MAT --model "$OUT/ft.model" \
  --sc 3 --mesh 21 --device cuda --out results/kappa/${MAT}_anharm.json
echo "[anharm] DONE -> results/kappa/${MAT}_anharm.json"
