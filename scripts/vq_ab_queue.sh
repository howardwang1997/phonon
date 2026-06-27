#!/bin/bash
# A->B autonomous queue on the V100:
#  A1 NbSe2 vc-relax (2Dxy) -> a_DFT, t_DFT
#  A2 NbSe2 fc2 at the relaxed geometry -> clean soft-mode magnitude
#  B  NbSe2 electronic bands -> k_F on Gamma-M -> 2k_F vs CDW q (2/3 GM)
set -uo pipefail
cd /root/phonon
export LD_LIBRARY_PATH=/root/miniconda3/envs/phonon/lib
PY=/root/miniconda3/envs/phonon/bin/python

echo "=== A1 NbSe2 vc-relax $(date) ==="
$PY scripts/vq3b_nbse2_relax.py --pw /root/gpupw.sh --nproc 1 --pseudo-dir pseudo \
  --a 3.44 --thickness 3.34 --ecutwfc 70 --ecutrho 280 --kpts 12 --degauss 0.015 \
  --workdir results/vq3b > /root/A1_relax.log 2>&1 || { echo "[queue] A1 FAILED"; exit 2; }
A_DFT=$(grep -oE "A_DFT=[0-9.]+" /root/A1_relax.log | head -1 | cut -d= -f2)
T_DFT=$(grep -oE "THICKNESS_DFT=[0-9.]+" /root/A1_relax.log | head -1 | cut -d= -f2)
A_DFT=${A_DFT:-3.44}; T_DFT=${T_DFT:-3.34}
echo "[queue] relaxed a=$A_DFT thickness=$T_DFT"

echo "=== A2 NbSe2 fc2 at relaxed geom $(date) ==="
$PY scripts/vq3_nbse2_dft.py --pw /root/gpupw.sh --nproc 1 --pseudo-dir pseudo \
  --a "$A_DFT" --thickness "$T_DFT" --supercell 3 --ecutwfc 70 --ecutrho 280 --kpts 6 \
  --degauss 0.015 --workdir results/vq3b --tag nbse2_dft_relaxed \
  > /root/A2_fc2.log 2>&1 || { echo "[queue] A2 FAILED"; exit 3; }

echo "=== B NbSe2 electronic bands / 2k_F $(date) ==="
$PY scripts/vq3c_nbse2_bands.py --pw /root/gpupw.sh --nproc 1 --pseudo-dir pseudo \
  --a "$A_DFT" --thickness "$T_DFT" --ecutwfc 70 --ecutrho 280 --nk 18 --npoints 200 \
  --workdir results/vq3c > /root/B_bands.log 2>&1 || echo "[queue] B FAILED"

echo "=== AB QUEUE DONE $(date) ==="
echo "--- A2 relaxed soft mode ---"; grep -E "min freq|SOFT MODE" /root/A2_fc2.log | tail -2
echo "--- B 2k_F ---"; grep -E "k_F|q_CDW" /root/B_bands.log | tail -6
