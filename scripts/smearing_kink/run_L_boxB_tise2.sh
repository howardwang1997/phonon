#!/usr/bin/env bash
# Box B (L)-channel PLAN-B: TiSe2 4x4 Path-P only.
# (TaS2/TaSe2 3x3 data reused from data/v100/path_p -> relayed to 2060; not re-run.)
set -u
PY=/root/miniconda3/envs/phonon/bin/python; cd /root/phonon
run_pp(){ N=$1; Y=$2; D=data/path_p_${N}
  [ -f "$D/train.xyz" ] && { echo "[B-pp] $N done"; exit 0; }
  echo "[B-pp] $N $(date +%T)"
  $PY scripts/path_p_nbse2_make_data.py --yaml "$Y" --pw /root/gpupw.sh \
    --pseudo-dir /root/phonon/pseudo --workdir results/path_p_${N} --outdir $D 2>&1 \
    | grep -E "pathP|saved|Error|wrote" | tail -3; }
run_pp 1T-TiSe2 results/v100/fc2_tise2_4x4_0.015/1T-TiSe2_phonopy.yaml
echo "[B] BOX B L-DONE (TiSe2) $(date)"
