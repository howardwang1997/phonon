#!/usr/bin/env bash
# Path-P NbSe2 Stage C: fine-tune the FOUNDATION MACE on the thermal/CDW-double-well
# DFT forces (data/path_p_nbse2). This is the *anharmonic* distillation -- contrast
# with the *harmonic* fc2 distillation (results/finetune_nbse2/ft_nbse2.model) which
# only saw small displacements and gave a too-shallow CDW well (SSCHA #1).
#
# Energy-aware (energy_weight 1.0) so the model learns the well DEPTH, not just the
# local curvature. Splits a 10% val set off train.xyz; test.xyz stays held-out.
#
# Usage:  bash scripts/finetune_pathp_nbse2.sh [epochs] [device]
set -e
EPOCHS="${1:-150}"
DEVICE="${2:-cuda}"
PY="${PHONON_PY:-$HOME/miniconda3/envs/phonon/bin/python}"
export LD_LIBRARY_PATH="$(cd "$(dirname "$PY")/.." && pwd)/lib:${LD_LIBRARY_PATH:-}"
MACE_TRAIN="$($PY -c 'import os,sys; p=os.path.join(os.path.dirname(sys.executable),"mace_run_train"); print(p if os.path.exists(p) else "mace_run_train")')"
DATA="${DATA_DIR:-data/path_p_nbse2}"
OUT="${OUT_DIR:-results/finetune_path_p_nbse2}"
NAME="${NAME:-ft}"
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# split 10% of train.xyz -> val.xyz (deterministic)
$PY - "$DATA" <<'PY'
import sys, numpy as np
from ase.io import read, write
d = sys.argv[1]
configs = read(f"{d}/train.xyz", ":")
# Path-P data has ABSOLUTE DFT energies (~-18627 eV); shift to relative so the
# E0s=0 / force-dominated recipe (proven for harmonic distillation) works.
emean = float(np.mean([float(a.info["REF_energy"]) for a in configs]))
for a in configs:
    a.info["REF_energy"] = float(a.info["REF_energy"]) - emean
print(f"[split] subtracted emean={emean:.1f} eV (energies now relative)")
rng = np.random.default_rng(0)
idx = rng.permutation(len(configs))
nval = max(2, int(round(0.10*len(configs))))
val = [configs[i] for i in idx[:nval]]
fit = [configs[i] for i in idx[nval:]]
write(f"{d}/train_fit.xyz", fit, format="extxyz")
write(f"{d}/val.xyz", val, format="extxyz")
print(f"[split] {len(fit)} fit + {len(val)} val (of {len(configs)})")
PY

E0S=$($PY -c "
from ase.io import read
zs = sorted({int(z) for a in read('$DATA/train.xyz', ':') for z in a.numbers})
print('{' + ','.join(f'{z}:0.0' for z in zs) + '}')
")
echo "pathP-nbse2 fine-tune: epochs=$EPOCHS device=$DEVICE E0s=$E0S data=$DATA out=$OUT"

"$MACE_TRAIN" \
  --name "$NAME" \
  --foundation_model small \
  --multiheads_finetuning False \
  --foundation_model_elements True \
  --scaling no_scaling \
  --save_cpu \
  --train_file "$DATA/train_fit.xyz" \
  --valid_file "$DATA/val.xyz" \
  --energy_key REF_energy \
  --forces_key REF_forces \
  --E0s "$E0S" \
  --energy_weight "${ENERGY_WEIGHT:-0.001}" \
  --forces_weight "${FORCES_WEIGHT:-100.0}" \
  --max_num_epochs "$EPOCHS" \
  --batch_size 4 \
  --valid_batch_size 4 \
  --eval_interval 5 \
  --lr 0.0001 \
  --default_dtype float32 \
  --device "$DEVICE" \
  --seed 1 \
  --model_dir "$OUT" \
  --checkpoints_dir "$OUT/checkpoints" \
  --log_dir "$OUT/logs" \
  --results_dir "$OUT/results"
echo "FT_PATHP_NBSE2_DONE -> $OUT/$NAME.model"
