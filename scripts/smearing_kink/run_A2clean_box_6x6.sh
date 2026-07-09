#!/usr/bin/env bash
# A2-CLEAN: 6×6 fc2 (multiple of 3 -> DOES represent the CDW q=(1/3,1/3), unlike 4×4).
# Harmonic convergence test: is the 3×3 soft mode (at q=1/3) converged, or truncated
# by the 3×3 ~10Å cutoff? 6×6 (~20Å cutoff) at the SAME q answers it cleanly.
# Compare 6×6 bare min-freq to 3×3 (NbS2 -1.90 THz; TaSe2 ~similar). If alike -> converged,
# the 3×3 (L)-inert story stands and the A2 "4×4 -264" caveat was a q-commensurability artifact.
# Usage: bash run_A2clean_box_6x6.sh <MaterialName>   (e.g. NbS2, 2H-TaSe2)
set -u
N="${1:?usage: $0 <MaterialName> e.g. NbS2}"
PY=/root/miniconda3/envs/phonon/bin/python
cd /root/phonon
SC=6
W=results/v100/fc2_${N}_${SC}x${SC}_0.015
Y=$W/${N}_phonopy.yaml
if [ -f "$Y" ]; then echo "[A2-6x6] $N ${SC}x${SC} fc2 already exists -> $Y"; exit 0; fi
C=/root/cfg_${N}_${SC}x${SC}.yaml
sed "s/^  fc2_supercell: 3/  fc2_supercell: ${SC}/" configs/v100_campaign.yaml > "$C"
echo "[A2-6x6] $N ${SC}x${SC} fc2 START $(date)"
$PY scripts/v100/tmd_dft_fc2.py --name "$N" --config "$C" --pw /root/gpupw.sh \
  --nproc 1 --pseudo-dir /root/phonon/pseudo --workdir "$W" 2>&1 \
  | grep -E "min freq|FAIL|SOFT|displaced supercell|atoms" | tail -4
echo "[A2-6x6] $N ${SC}x${SC} fc2 DONE $(date) -> $Y"
touch "$W/.6x6_done"
