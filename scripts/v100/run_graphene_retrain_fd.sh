#!/usr/bin/env bash
# Retrain graphene FC-distillation backbone on the fd dg0.08 fc2 (relaxed a=2.4576),
# matching the v11 architecture (r_max=5, 2 interactions, 64x0e+64x1o), on V100 Box B.
# 16 FC seeds -> ~2800 configs; from-scratch MACE (no foundation). Run in tmux ON the box.
set -uo pipefail
cd "$HOME/phonon"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate phonon
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
PY="$HOME/miniconda3/envs/phonon/bin/python"
mkdir -p data/finetune_graphene_fd results/gr_backbone_fd
LOG="results/gr_backbone_fd/retrain.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== graphene backbone retrain (fd, v11 arch) start $(date) ==="
FC2="results/graphene_kohn_fd/graphene_sc6_dg0.08_phonopy.yaml"
[ -f "$FC2" ] || { echo "!! missing $FC2"; exit 1; }
# 1. FC-distill data: 16 seeds x ~160 rattled configs (skip if already generated)
if [ ! -f data/finetune_graphene_fd/train.xyz ]; then
  echo "--- generate FC-distill data (16 seeds) ---"
  for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    "$PY" scripts/m1_1b_make_graphene_data.py --phonopy "$FC2" \
         --out "data/finetune_graphene_fd/seed$s" --n-configs 160 --seed "$s" --rattle-std 0.04 \
         > /dev/null 2>&1 || echo "!! seed $s data-gen partial"
  done
  cat data/finetune_graphene_fd/seed*/train.xyz > data/finetune_graphene_fd/train.xyz
  cat data/finetune_graphene_fd/seed*/val.xyz   > data/finetune_graphene_fd/val.xyz
fi
echo "train configs: $(grep -c 'Lattice' data/finetune_graphene_fd/train.xyz 2>/dev/null)  val: $(grep -c 'Lattice' data/finetune_graphene_fd/val.xyz 2>/dev/null)"
# 2. train MACE (v11 arch)
echo "--- mace_run_train (v11 arch) ---"
"$PY" -m mace.cli.run_train --config "configs/graphene_backbone_fd.yml" \
      --name graphene_backbone_fd \
      --work_dir results/gr_backbone_fd --log_dir results/gr_backbone_fd --device cuda \
      --checkpoints_dir results/gr_backbone_fd/checkpoints
echo "=== graphene backbone retrain DONE $(date) ==="
touch results/gr_backbone_fd/RETRAIN_DONE
