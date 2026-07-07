#!/usr/bin/env bash
# B1: VSe2 MD-TDEP through T_CDW=110K — the overdamped (L)-spectrum SSCHA cannot give.
# MLIP-MD at 6 T (50/80/110/150/200/300K, spanning the 110K CDW transition) on the
# validated VSe2 Path-P FT model -> fit effective fc2(T)+fc3 -> omega(T) dispersion + linewidth.
set -u
cd ~/phonon
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}
PY=$HOME/miniconda3/envs/phonon/bin/python
MODEL=results/finetune_path_p_1T-VSe2/ft.model
[ -f "$MODEL" ] || { echo "[B1] MISSING model $MODEL"; exit 1; }

# Extract VSe2 primitive (3-atom 1T cell) from the fc2 yaml -> xyz for td_anharmonic --structure
STRUC=/tmp/VSe2_prim.xyz
if [ ! -f "$STRUC" ]; then
  $PY - <<'PYEOF' 2>&1 | grep -vE "Warning|warn" | tail -3
import sys; sys.path.insert(0, "scripts/smearing_kink"); sys.path.insert(0, "src")
import friedel_module as fm
from phonon_accel.phonons import phonopy_to_ase
from ase.io import write
ph = fm.load_ph("results/vq_family/vse2/1T-VSe2_dg0.020.yaml")
write("/tmp/VSe2_prim.xyz", phonopy_to_ase(ph.primitive))
print("wrote /tmp/VSe2_prim.xyz")
PYEOF
fi
[ -f "$STRUC" ] || { echo "[B1] structure extraction FAILED"; exit 1; }

echo "[B1] VSe2 MD-TDEP START $(date +%T)  model=$MODEL"
$PY scripts/td_anharmonic.py --structure "$STRUC" --model "$MODEL" \
  --tag 1T-VSe2_tdep --temperatures 50,80,110,150,200,300 \
  --supercell 4,4,1 --device cuda --outdir results/td_phonon 2>&1 \
  | grep -E "T=|Kink|kink|minfreq|fc3|anharm|Error|Traceback|wrote|DONE|FAIL" | tail -40
echo "[B1] VSe2 TDEP DONE $(date)"
