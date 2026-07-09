#!/usr/bin/env bash
# Re-distill the graphene backbone (8x8 dg0.080, more configs + epochs) to fix the
# acoustic sound-velocity artifact of the old gr_backbone (6x6, 80ep).
# Box B (make_data, V100 QE) -> relay 2060 -> finetune (2060 GPU).
set -uo pipefail
PY=/root/miniconda3/envs/phonon/bin/python
cd /root/phonon
R2060=howardwang@100.105.21.7
FC2=results/sc_conv/graphene_sc8_dg0.08_phonopy.yaml
DATA=data/finetune_graphene8

echo "[redistill] make_data (8x8 dg0.080) START $(date)"
if [ ! -f "$DATA/train.xyz" ]; then
  $PY scripts/m1_1b_make_graphene_data.py --phonopy "$FC2" --out "$DATA" --n-configs 100 2>&1 | tail -5
fi
echo "[redistill] relay data -> 2060 $(date)"
ssh -o ControlPath=none -o StrictHostKeyChecking=no "$R2060" "mkdir -p ~/phonon/$DATA" 2>/dev/null
for f in train.xyz val.xyz test.xyz; do [ -f "$DATA/$f" ] && scp -o ControlPath=none -q "$DATA/$f" "$R2060:~/phonon/$DATA/" 2>/dev/null; done
ssh -o ControlPath=none -o StrictHostKeyChecking=no "$R2060" "touch ~/phonon/$DATA/.relayed" 2>/dev/null
echo "[redistill] relayed. finetune runs on 2060 via run_graphene_redistill_2060.sh (launched separately)."
echo "[redistill] Box B DONE $(date)"
