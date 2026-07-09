#!/usr/bin/env bash
# M4 (2060): B2d 2D kink(T_el,T_lat) surface. WAITS for M1 (2060) to finish, then
# smoke-tests 1 SSCHA point; if sane, runs the full (T_el,T_lat) grid.
set -uo pipefail
cd ~/phonon
PY=/home/howardwang/miniconda3/envs/phonon/bin/python
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BB=results/vq_family/vse2/1T-VSe2_dg0.020.yaml
TM=results/vq_family/vse2/1T-VSe2_dg0.005.yaml
MD=results/finetune_path_p_1T-VSe2/ft.model
DONE=results/td_phonon/.M1_2060_all_done

echo "[M4] $(date) waiting for M1 (2060) to finish (.M1_2060_all_done)..."
while [ ! -f "$DONE" ]; do sleep 120; done
echo "[M4] $(date) M1 2060 done -> GPU free -> B2d 2D SSCHA"

echo "[M4] smoke-test 1 point (T_el=789, T_lat=20) $(date +%T)"
$PY scripts/smearing_kink/b2d_2d_sscha.py --backbone "$BB" --template "$TM" --model "$MD" \
  --tag b2d_2d_smoke --tel 789 --tlat 20 --nconfigs 100 --maxpop 3 --device cuda 2>&1 | tail -8
SM=results/smearing_kink/b2d_2d_smoke.csv
if [ ! -f "$SM" ] || [ "$(wc -l < "$SM")" -lt 2 ]; then
  echo "[M4] SMOKE-TEST FAILED (no csv) -> abort"; touch results/td_phonon/.M4_failed; exit 1
fi
echo "[M4] smoke OK -> full 3x3 grid $(date +%T)"
$PY scripts/smearing_kink/b2d_2d_sscha.py --backbone "$BB" --template "$TM" --model "$MD" \
  --tag b2d_2d_vse2 --tel 789,2368,4737 --tlat 20,110,200 --nconfigs 200 --maxpop 4 --device cuda 2>&1 | tail -25
touch results/td_phonon/.M4_done
echo "[M4] DONE $(date)"
