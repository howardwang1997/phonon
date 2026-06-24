#!/usr/bin/env bash
# M1.1b step 3: graphene-specific FC distillation -- same single-head recipe as
# the general FT (scripts/finetune_mace_gpu.sh) but on graphene's own DFT fc2
# data, so any Gamma/K-cusp recovery is attributable to the graphene-specific
# curvature, not the procedure.
#
# Usage:  bash scripts/finetune_graphene.sh [epochs] [device]
set -e
EPOCHS="${1:-80}"
DEVICE="${2:-cuda}"
PY="${PHONON_PY:-$HOME/miniconda3/envs/phonon/bin/python}"
# conda env's libstdc++ has the CXXABI the pip/conda compiled extensions need;
# force the loader to it (system libstdc++ is older).
export LD_LIBRARY_PATH="$(cd "$(dirname "$PY")/.." && pwd)/lib:${LD_LIBRARY_PATH:-}"
MACE_TRAIN="$($PY -c 'import os,shutil,sys; p=os.path.join(os.path.dirname(sys.executable),"mace_run_train"); print(p if os.path.exists(p) else "mace_run_train")')"
DATA="${DATA_DIR:-data/finetune_graphene}"
OUT="${OUT_DIR:-results/finetune_graphene}"
mkdir -p "$OUT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E0S=$($PY -c "
from ase.io import read
zs = sorted({int(z) for a in read('$DATA/train.xyz', ':') for z in a.numbers})
print('{' + ','.join(f'{z}:0.0' for z in zs) + '}')
")
echo "graphene fine-tune: epochs=$EPOCHS device=$DEVICE E0s=$E0S data=$DATA"

"$MACE_TRAIN" \
  --name "${NAME:-ft_graphene}" \
  --foundation_model small \
  --multiheads_finetuning False \
  --foundation_model_elements True \
  --train_file "$DATA/train.xyz" \
  --valid_file "$DATA/val.xyz" \
  --energy_key REF_energy \
  --forces_key REF_forces \
  --E0s "$E0S" \
  --energy_weight "${ENERGY_WEIGHT:-0.01}" \
  --forces_weight 100.0 \
  --max_num_epochs "$EPOCHS" \
  --batch_size 4 \
  --valid_batch_size 4 \
  --eval_interval 4 \
  --lr 0.001 \
  --default_dtype float32 \
  --device "$DEVICE" \
  --seed 1 \
  --model_dir "$OUT" \
  --checkpoints_dir "$OUT/checkpoints" \
  --log_dir "$OUT/logs" \
  --results_dir "$OUT/results"
echo "FT_GRAPHENE_DONE -> $OUT/${NAME:-ft_graphene}.model"
