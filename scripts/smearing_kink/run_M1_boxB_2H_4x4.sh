#!/usr/bin/env bash
# M1: 2H 4×4 recheck (resolve A2 caveat). For each 2H-CDW material, make 4×4 fc2 +
# 4×4 Path-P data, then relay BOTH to 2060 (which fine-tunes + SSCHAs).
# Tests whether 2H (L)-inert (3×3 ~0) holds at 4×4 — A2 found NbS2 4×4 gives -264 cm^-1
# (3×3 ~0). NbSe2/TaS2/TaSe2 are the 2H-CDW materials (NbS2 has no CDW).
# Box B (V100 DFT) -> 2060 (fine-tune + SSCHA). ~4h DFT + ~6h Path-P = ~10-12h total.
set -u
PY=/root/miniconda3/envs/phonon/bin/python
cd /root/phonon
SC=4
R2060=howardwang@100.105.21.7
PSEUDO=/root/phonon/pseudo

for N in 2H-TaSe2 NbSe2 2H-TaS2; do
  echo "========================================"
  echo "[M1] $N 4x4  START $(date)"
  W=results/v100/fc2_${N}_${SC}x${SC}_0.015
  Y=$W/${N}_phonopy.yaml
  # 1. 4x4 fc2 (dg0.015)
  if [ ! -f "$Y" ]; then
    C=/root/cfg_${N}_4x4.yaml
    sed "s/^  fc2_supercell: 3/  fc2_supercell: ${SC}/" configs/v100_campaign.yaml > "$C"
    echo "[M1-fc2] $N 4x4 fc2 START $(date +%T)"
    $PY scripts/v100/tmd_dft_fc2.py --name "$N" --config "$C" --pw /root/gpupw.sh \
      --nproc 1 --pseudo-dir "$PSEUDO" --workdir "$W" 2>&1 | grep -E "min freq|FAIL|SOFT|Error" | tail -3
  else
    echo "[M1-fc2] $N 4x4 fc2 exists, skip"
  fi
  # 2. 4x4 Path-P data
  D=data/path_p_${N}_4x4
  if [ ! -f "$D/train.xyz" ]; then
    echo "[M1-pp] $N 4x4 Path-P START $(date +%T)"
    $PY scripts/path_p_nbse2_make_data.py --yaml "$Y" --pw /root/gpupw.sh \
      --pseudo-dir "$PSEUDO" --workdir results/path_p_${N}_4x4 --outdir "$D" 2>&1 \
      | grep -E "pathP|saved|Error|wrote|cfg" | tail -3
  else
    echo "[M1-pp] $N 4x4 Path-P exists, skip"
  fi
  # 3. relay path_p data + 4x4 fc2 yaml -> 2060
  if [ -f "$D/train.xyz" ]; then
    ssh -o ControlPath=none -o StrictHostKeyChecking=no "$R2060" "mkdir -p ~/phonon/$D ~/phonon/$W" 2>/dev/null
    for f in train.xyz test.xyz all.xyz; do
      [ -f "$D/$f" ] && scp -o ControlPath=none -q "$D/$f" "$R2060:~/phonon/$D/" 2>/dev/null; done
    [ -f "$Y" ] && scp -o ControlPath=none -q "$Y" "$R2060:~/phonon/$W/" 2>/dev/null
    ssh -o ControlPath=none -o StrictHostKeyChecking=no "$R2060" "touch ~/phonon/$D/.M1_relayed" 2>/dev/null
    echo "[M1-relay] $N 4x4 -> 2060 DONE $(date +%T)"
  else
    echo "[M1-relay] $N 4x4 NO train.xyz, SKIP relay (check errors above)"
  fi
  echo "[M1] $N 4x4  DONE $(date)"
done
echo "========================================"
echo "[M1] BOX B 2H-4x4 ALL DONE $(date)"
