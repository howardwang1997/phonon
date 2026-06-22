#!/usr/bin/env bash
# GPU-side of the converged-fc3 curvature re-distillation: take pre-generated
# curvature data (relayed from a P4 box that produced the CONVERGED fc3) and
# fine-tune br_B16 on it, then eval Si κ at sc3+sc4. Pairs with the orchestration
# watcher that relays $DATA from a fresh box once its converged fc3 is ready.
#   DATA=data/curv_conv3 SUF=_conv3 GPU=1 bash scripts/finetune_curvature.sh
set -uo pipefail
cd ~/phonon
PY=~/miniconda3/envs/phonon/bin/python
MT=~/miniconda3/envs/phonon/bin/mace_run_train
DATA=${DATA:-data/curv_Si}; SUF=${SUF:-_conv}; GPU=${GPU:-1}
BASE=results/ablation/br_B16_s1/ft.model
E0S=$($PY -c "from ase.io import read; zs=sorted({int(z) for a in read('$DATA/train.xyz',':') for z in a.numbers}); print('{'+','.join(f'{z}:0.0' for z in zs)+'}')")
OUT=results/ablation/curv_Si$SUF; rm -rf "$OUT"; mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $MT --name ft --foundation_model "$BASE" \
  --multiheads_finetuning True --num_samples_pt 1000 --subselect_pt random --foundation_model_elements True \
  --train_file $DATA/train.xyz --valid_file $DATA/val.xyz \
  --energy_key REF_energy --forces_key REF_forces --E0s "$E0S" \
  --energy_weight 0.1 --forces_weight 100.0 --max_num_epochs 80 \
  --batch_size 16 --valid_batch_size 16 --eval_interval 20 --lr 0.0005 \
  --default_dtype float32 --device cuda --seed 1 \
  --model_dir "$OUT" --results_dir "$OUT" --log_dir "$OUT" --checkpoints_dir "$OUT" --save_cpu

for sc in 3 4; do
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mlip_kappa.py --material Si --model "$OUT/ft.model" \
    --sc $sc --mesh 21 --device cuda --out results/kappa/Si_curv${SUF}_sc${sc}.json
done
echo "[curv$SUF] DONE -> compare Si_curv${SUF}_sc4 to baseline45 / harmFT143 / sc2-curv47"
