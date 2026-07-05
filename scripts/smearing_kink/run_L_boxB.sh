#!/usr/bin/env bash
set -u
PY=/root/miniconda3/envs/phonon/bin/python; cd /root/phonon
run_fc2(){ N=$1; SC=$2; W=results/v100/fc2_${N}_3x3_0.015
  [ -f "$W"/*_phonopy.yaml ] 2>/dev/null && return 0
  C=/root/cfg_${N}_0.015.yaml
  sed "s/^  fc2_supercell: 3/  fc2_supercell: ${SC%%,*}/" configs/v100_campaign.yaml > "$C"
  echo "[B-fc2] $N $(date +%T)"
  $PY scripts/v100/tmd_dft_fc2.py --name $N --config "$C" --pw /root/gpupw.sh --nproc 1 --pseudo-dir pseudo --workdir "$W" 2>&1 | grep -E "min freq|FAIL" | tail -2; }
run_pp(){ N=$1; Y=$2; D=data/path_p_${N}
  [ -f "$D/train.xyz" ] && { echo "[B-pp] $N done"; return 0; }
  echo "[B-pp] $N $(date +%T)"
  $PY scripts/path_p_nbse2_make_data.py --yaml "$Y" --pw /root/gpupw.sh --pseudo-dir pseudo --workdir results/path_p_${N} --outdir $D 2>&1 | grep -E "pathP|saved|Error|wrote" | tail -3; }
run_fc2 2H-TaS2 3,3,1
run_pp 2H-TaS2 results/v100/fc2_2H-TaS2_3x3_0.015/2H-TaS2_phonopy.yaml
run_pp 2H-TaSe2 results/v100/fc2_tase2_3x3_0.015/2H-TaSe2_phonopy.yaml
run_pp 1T-TiSe2 results/v100/fc2_tise2_4x4_0.015/1T-TiSe2_phonopy.yaml
echo "[B] BOX B L-DONE $(date)"
