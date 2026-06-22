#!/usr/bin/env bash
# P2: curvature-aware 3rd-order distillation. Joint harmonic-anchor + anharmonic
# data (preserves Φ₂ Hessian) + replay fine-tune from the harmonic-FT model.
# Success = Si κ at the CONVERGED supercell (sc4) stays ≈140 (does NOT regress to
# ~45 like the naive 3rd-order did) while Φ₃ info is present.
set -uo pipefail
cd ~/phonon
PY=~/miniconda3/envs/phonon/bin/python
MT=~/miniconda3/envs/phonon/bin/mace_run_train
MAT=Si
BASE=results/ablation/br_B16_s1/ft.model
GPU=${GPU:-0}

$PY scripts/make_curvature_data.py --material $MAT --fc-dir results/fc3 --out data/curv_$MAT \
  --n-harm 450 --n-anharm 150 --harm-weight 3.0
E0S=$($PY -c "from ase.io import read; zs=sorted({int(z) for a in read('data/curv_${MAT}/train.xyz',':') for z in a.numbers}); print('{'+','.join(f'{z}:0.0' for z in zs)+'}')")
OUT=results/ablation/curv_$MAT; rm -rf "$OUT"; mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $MT --name ft --foundation_model "$BASE" \
  --multiheads_finetuning True --num_samples_pt 1000 --subselect_pt random \
  --foundation_model_elements True \
  --train_file data/curv_${MAT}/train.xyz --valid_file data/curv_${MAT}/val.xyz \
  --energy_key REF_energy --forces_key REF_forces --E0s "$E0S" \
  --energy_weight 0.1 --forces_weight 100.0 --max_num_epochs 80 \
  --batch_size 16 --valid_batch_size 16 --eval_interval 20 --lr 0.0005 \
  --default_dtype float32 --device cuda --seed 1 \
  --model_dir "$OUT" --results_dir "$OUT" --log_dir "$OUT" --checkpoints_dir "$OUT" --save_cpu

for sc in 3 4; do
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mlip_kappa.py --material $MAT --model "$OUT/ft.model" \
    --sc $sc --mesh 21 --device cuda --out results/kappa/Si_curv_sc${sc}.json
done
echo "[curvature] DONE  (compare Si_curv_sc4 to baseline45 / harmFT143 / naive-anharm42)"
