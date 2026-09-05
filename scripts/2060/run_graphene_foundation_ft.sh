#!/usr/bin/env bash
# Foundation fine-tune MACE-MP-0 (small) on graphene FC-distillation data on 2060.
# ABLATION: does foundation smooth the Kohn anomaly vs FC-distillation?
# batch_size=1 (2060 8GB — foundation model + force double-backprop is VRAM-heavy).
set -euo pipefail
cd "$HOME/phonon"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
CONDA="$HOME/miniconda3/bin/conda"
mkdir -p data/finetune_graphene_fd results/gr_foundation_ft/checkpoints
LOG="results/gr_foundation_ft/ft.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== foundation fine-tune (graphene Kohn ablation) start $(date) ==="
# 1. gen data if not present
if [ ! -f data/finetune_graphene_fd/train.xyz ]; then
  echo "--- gen FC-distill data (16 seeds) ---"
  FC2="results/graphene_kohn_fd/graphene_sc6_dg0.08_phonopy.yaml"
  for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    "$CONDA" run --no-capture-output -n phonon python scripts/m1_1b_make_graphene_data.py --phonopy "$FC2" \
         --out "data/finetune_graphene_fd/seed$s" --n-configs 160 --seed "$s" --rattle-std 0.04 \
         > /dev/null
  done
  cat data/finetune_graphene_fd/seed*/train.xyz > data/finetune_graphene_fd/train.xyz
  cat data/finetune_graphene_fd/seed*/val.xyz   > data/finetune_graphene_fd/val.xyz
fi
echo "train configs: $(grep -c 'Lattice' data/finetune_graphene_fd/train.xyz)"
# 2. foundation fine-tune (batch_size=1 for 8GB VRAM)
MODEL="results/gr_foundation_ft/gr_foundation_ft.model"
if [[ ! -s "$MODEL" ]]; then
  echo "--- finalize the best foundation fine-tune checkpoint ---"
  CHECKPOINT="$(find results/gr_foundation_ft/checkpoints -type f -name 'gr_foundation_ft_run-*_epoch-*.pt' | sort -V | tail -1)"
  EXTRA=()
  if [[ -n "$CHECKPOINT" ]]; then
    CKPT_EPOCH="${CHECKPOINT##*_epoch-}"
    CKPT_EPOCH="${CKPT_EPOCH%.pt}"
    echo "resume checkpoint: $CHECKPOINT (epoch $CKPT_EPOCH); no discarded epochs will be repeated"
    EXTRA+=(--restart_latest --max_num_epochs "$CKPT_EPOCH")
  fi
  "$CONDA" run --no-capture-output -n phonon python -m mace.cli.run_train \
        --config "configs/graphene_foundation_ft.yml" --name gr_foundation_ft \
        --work_dir results/gr_foundation_ft --log_dir results/gr_foundation_ft \
        --results_dir results/gr_foundation_ft/results \
        --model_dir results/gr_foundation_ft \
        --checkpoints_dir results/gr_foundation_ft/checkpoints \
        --device cuda --save_cpu --plot False "${EXTRA[@]}"
fi
[[ -s "$MODEL" ]]

echo "--- deploy fine-tuned foundation + healing term ---"
EVAL="results/gr_foundation_ft/deploy_healing.csv"
"$CONDA" run --no-capture-output -n phonon python \
  scripts/smearing_kink/deploy_foundation_initial.py \
  --model "$MODEL" --device cuda --default-dtype float32 \
  --label foundation_finetuned --output "$EVAL"
[[ -s "$EVAL" ]]
"$CONDA" run -n phonon python -c '
import csv, math, sys
with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 9
assert all(math.isfinite(float(row["abs_error"])) for row in rows)
' "$EVAL"

touch results/gr_foundation_ft/FT_DONE
echo "=== foundation FT RECOVERY VERIFIED $(date) ==="
