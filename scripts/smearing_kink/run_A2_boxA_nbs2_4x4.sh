#!/usr/bin/env bash
# A2: NbS2 4x4 Path-P convergence test — is the 2H "~0" (L)-instability a 3x3 artifact?
# Pipeline: NbS2 4x4 fc2 (dg0.015) -> path_p make_data (4x4). Then relay to 2060 for FT+SSCHA.
# NbS2 = deepest bare soft mode (-103 cm-1) among 2H -> the strongest test.
set -u
cd /root/phonon
PY=/root/miniconda3/envs/phonon/bin/python
SC=4
# 1. NbS2 4x4 fc2
W=results/v100/fc2_NbS2_${SC}x${SC}_0.015
if ! ls "$W"/NbS2_phonopy.yaml >/dev/null 2>&1; then
  sed "s/^  fc2_supercell: 3/  fc2_supercell: ${SC}/" configs/v100_campaign.yaml > /root/cfg_NbS2_${SC}x${SC}.yaml
  echo "[A2] NbS2 ${SC}x${SC} fc2 START $(date +%T)"
  $PY scripts/v100/tmd_dft_fc2.py --name NbS2 --config /root/cfg_NbS2_${SC}x${SC}.yaml \
    --pw /root/gpupw.sh --nproc 1 --pseudo-dir /root/phonon/pseudo --workdir "$W" 2>&1 \
    | grep -E "min freq|FAIL|SOFT|Error|JOB DONE" | tail -3
fi
# 2. path_p make_data on 4x4 fc2
D=data/path_p_NbS2_${SC}x${SC}
if [ ! -f "$D/train.xyz" ]; then
  echo "[A2] NbS2 ${SC}x${SC} path_p START $(date +%T)"
  $PY scripts/path_p_nbse2_make_data.py --yaml "$W"/NbS2_phonopy.yaml \
    --pw /root/gpupw.sh --pseudo-dir /root/phonon/pseudo \
    --workdir results/path_p_NbS2_${SC}x${SC} --outdir "$D" 2>&1 \
    | grep -E "pathP|saved|Error|wrote|cfg" | tail -3
fi
echo "[A2] NbS2 ${SC}x${SC} PATH-P DONE $(date)"
