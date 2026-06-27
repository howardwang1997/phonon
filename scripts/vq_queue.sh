#!/bin/bash
# Autonomous V100 single-card DFT queue: waits for V-Q1, then V-Q2 (E-channel
# degauss scan, Fermi-Dirac so degauss = k_B*T_el), then V-Q3 (NbSe2 Gate#2).
# Usage: nohup bash /root/vq_queue.sh <VQ1_PID> > /root/vq_queue.log 2>&1 &
set -uo pipefail
cd /root/phonon
export LD_LIBRARY_PATH=/root/miniconda3/envs/phonon/lib
PY=/root/miniconda3/envs/phonon/bin/python
WAITPID=${1:-}

if [ -n "$WAITPID" ]; then
  echo "[queue] waiting for V-Q1 pid $WAITPID to release GPU ... $(date)"
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 30; done
  echo "[queue] V-Q1 finished $(date)"
fi

# ---- V-Q2: (E)-channel electronic-temperature scan (fixed 5x5 geometry) ----
# Fermi-Dirac smearing: degauss[Ry] -> T_el = degauss * 157887 K
for dg in 0.005 0.01 0.02 0.04; do
  echo "[queue] === V-Q2 fermi-dirac degauss=$dg ($(awk "BEGIN{printf \"%.0f\", $dg*157887}") K) $(date) ==="
  $PY scripts/m1_1b_graphene_dft.py --pw /root/gpupw.sh --nproc 1 \
    --pseudo-dir pseudo --pseudo C_ONCV_PBE-1.2.upf \
    --a 2.46 --supercell 5 --ecutwfc 60 --ecutrho 240 --kpts 6 \
    --smearing fermi-dirac --degauss "$dg" \
    --workdir results/vq2 --tag "graphene_dg${dg}" \
    > "/root/vq2_dg${dg}.log" 2>&1 && echo "[queue] V-Q2 dg=$dg OK" \
    || echo "[queue] V-Q2 dg=$dg FAILED"
done

# ---- V-Q3: NbSe2 CDW Gate#2 (3x3 supercell, metallic, dense k) ----
echo "[queue] === V-Q3 NbSe2 Gate#2 $(date) ==="
$PY scripts/vq3_nbse2_dft.py --pw /root/gpupw.sh --nproc 1 \
  --pseudo-dir pseudo --supercell 3 --ecutwfc 70 --ecutrho 280 --kpts 6 --degauss 0.015 \
  --workdir results/vq3 --tag nbse2_dft \
  > /root/vq3_nbse2.log 2>&1 && echo "[queue] V-Q3 OK" || echo "[queue] V-Q3 FAILED"

echo "[queue] ALL DONE $(date)"
