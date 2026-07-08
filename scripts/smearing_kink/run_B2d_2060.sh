#!/usr/bin/env bash
# B2d (2060): unified MLIP+LR -> kink(T_el, T_lat) for VSe2.
# WAITS for A2 (NbS2 4x4) to finish (frees the 8GB GPU), then runs the (E)-slice
# through the Path-P VSe2 backbone + Friedel and the unified reduced-T collapse
# vs the (L) SSCHA data. ~5 min GPU.
set -uo pipefail
cd ~/phonon
PY=/home/howardwang/miniconda3/envs/phonon/bin/python
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DONE=results/td_phonon/.A2_4x4_done

echo "[B2d] $(date) waiting for A2 4x4 to finish..."
while [ ! -f "$DONE" ]; do sleep 60; done
echo "[B2d] $(date) A2 done -> GPU free -> run B2d unified kink (VSe2)"
$PY scripts/smearing_kink/b2d_unified_kink.py \
  --backbone results/vq_family/vse2/1T-VSe2_dg0.020.yaml \
  --template results/vq_family/vse2/1T-VSe2_dg0.005.yaml \
  --model results/finetune_path_p_1T-VSe2/ft.model \
  --tag b2d_vse2 --device cuda 2>&1 | tee /tmp/B2d_vse2.log
touch results/td_phonon/.B2d_vse2_done
echo "[B2d] DONE $(date)"
