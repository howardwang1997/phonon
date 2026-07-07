#!/usr/bin/env bash
# Box A (E)-breadth: degauss-scan fc2 (3x3) for prior-planning gpu-lane materials not yet fc2'd.
#   1T-TaS2 (strongest CDW anomaly), 1T-TiS2 (metallic non-CDW control), MoS2 (gapped control).
# Each at degauss 0.005/0.010/0.015/0.020 -> extends (E)-family DOS->T1/2 law + gapped contrast
#   (MoS2 should show NO Friedel kink -> proves the (E)-kink is metallic-screening-driven).
# Idempotent: skips any (material,degauss) whose phonopy.yaml already exists.
set -u
PY=/root/miniconda3/envs/phonon/bin/python; cd /root/phonon
run_fc2(){ N=$1; DG=$2; SC=3; W=results/v100/fc2_${N}_${SC}x${SC}_${DG}
  if ls "$W"/*_phonopy.yaml >/dev/null 2>&1; then echo "[A-fc2] $N dg$DG done"; return 0; fi
  C=/root/cfg_${N}_${DG}.yaml
  # match only the fc2-section degauss line (value 0.015), not the relax-section 0.02
  sed "s/^  degauss: 0\.015/  degauss: ${DG}/;s/^  fc2_supercell: 3/  fc2_supercell: ${SC}/" \
    configs/v100_campaign.yaml > "$C"
  echo "[A-fc2] $N dg$DG $(date +%T)"
  $PY scripts/v100/tmd_dft_fc2.py --name "$N" --config "$C" --pw /root/gpupw.sh --nproc 1 \
    --pseudo-dir /root/phonon/pseudo --workdir "$W" 2>&1 | grep -E "min freq|FAIL|SOFT|Error" | tail -2; }
for N in 1T-TaS2 1T-TiS2 MoS2; do
  for DG in 0.005 0.010 0.015 0.020; do run_fc2 "$N" "$DG"; done
done
echo "[A] BOX A (E)-BREADTH DONE $(date)"
