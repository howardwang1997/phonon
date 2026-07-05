#!/usr/bin/env bash
# Box A (L)-channel: VSe2 fc2 (prereq) -> VSe2 Path-P -> NbS2 Path-P
set -u
PY=/root/miniconda3/envs/phonon/bin/python
CD=/root/phonon; cd $CD
PP=$PY  # DFT uses phonon env
run_fc2(){ N=$1; SC=$2; W=results/v100/fc2_${N}_$(echo $SC|tr ',' '_')_0.015
  [ -f "$W/*_phonopy.yaml" ] 2>/dev/null && return 0
  C=/root/cfg_${N}_0.015.yaml
  sed "s/^  degauss: 0.015/  degauss: 0.015/;s/^  fc2_supercell: 3/  fc2_supercell: ${SC%%,*}/" configs/v100_campaign.yaml > "$C"
  echo "[A-fc2] $N $(date +%T)"
  $PP scripts/v100/tmd_dft_fc2.py --name $N --config "$C" --pw /root/gpupw.sh --nproc 1 --pseudo-dir pseudo --workdir "$W" 2>&1 | grep -E "min freq|FAIL" | tail -2; }
run_pp(){ N=$1; Y=$2; D=data/path_p_${N}
  [ -f "$D/train.xyz" ] && { echo "[A-pp] $N done"; return 0; }
  echo "[A-pp] $N $(date +%T)"
  $PP scripts/path_p_nbse2_make_data.py --yaml "$Y" --pw /root/gpupw.sh --pseudo-dir pseudo --workdir results/path_p_${N} --outdir $D 2>&1 | grep -E "pathP|saved|Error|wrote" | tail -3; }
# VSe2: fc2 (4x4 for 1T) then Path-P
run_fc2 1T-VSe2 4,4,1
run_pp 1T-VSe2 results/v100/fc2_1T-VSe2_4_4_1_0.015/1T-VSe2_phonopy.yaml
# NbS2: Path-P (fc2 already exists)
run_pp NbS2 results/v100/fc2_nbs2_3x3_0.015/NbS2_phonopy.yaml
echo "[A] BOX A L-DONE $(date)"
