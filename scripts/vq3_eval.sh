#!/bin/bash
# V-Q3 Gate#2 decisive eval: NbSe2 MLIP dispersion at the SAME geometry/supercell
# as the DFT (a=3.44, 3x3x1, no-relax). Foundation (small) vs distilled-FT.
# Question: does the distilled MLIP recover the CDW soft mode (min freq << 0)
# that the foundation misses (min freq ~ 0)?
set -uo pipefail
cd /root/phonon
export LD_LIBRARY_PATH=/root/miniconda3/envs/phonon/lib
PY=/root/miniconda3/envs/phonon/bin/python
WAITPID=${1:-}
if [ -n "$WAITPID" ]; then
  echo "[eval] waiting for FT pid $WAITPID ... $(date)"
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 20; done
fi
test -f results/finetune_nbse2/ft_nbse2.model || { echo "[eval] FT model MISSING $(date)"; exit 1; }
echo "[eval] FT done; running Gate#2 dispersion eval $(date)"
for cfg in "small|nbse2_base" "results/finetune_nbse2/ft_nbse2.model|nbse2_ft"; do
  m="${cfg%%|*}"; tag="${cfg##*|}"
  for SC in "3,3,1" "6,6,1"; do
    s=$(echo "$SC" | tr -d ,)
    echo "[eval] === ${tag}_sc${s:0:1} (model=$m, sc=$SC) $(date) ==="
    $PY scripts/harmonic_dispersion_2d.py --structure data/td_phonon/nbse2.xyz \
      --model "$m" --model-type mace --tag "${tag}_sc${s:0:1}" \
      --supercell "$SC" --a 3.44 --no-relax --path GMKG --device cpu --outdir results/vq3 \
      2>&1 | grep -E "min freq|top optical|NO-RELAX|rebuilt" || echo "[eval] $tag sc=$SC FAILED"
  done
done
echo "[eval] DONE $(date)"
