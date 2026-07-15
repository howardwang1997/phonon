#!/usr/bin/env bash
# Re-run the 4 TMD fc2 at dg0.002 with Fermi-Dirac smearing (physically-correct
# T_el: degauss = k_B*T_el exactly; dg0.002 -> T_el=316 K). Replaces the cold-
# smearing comparison set with an fd-based true-0K set. Distinct _fd dirs (the
# cold dg0.002 yamls are kept). Run in tmux ON the box.
#   A: NbSe2 NbS2 (3x3)   B: 2H-TaSe2 (3x3) 1T-VSe2 (4x4)
set -uo pipefail
BOX="${1:?need A|B}"
cd "$HOME/phonon"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate phonon
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="$HOME/miniconda3/envs/phonon/bin/python"
mkdir -p results/v100/fc2_fd
LOG="results/v100/fc2_fd/run_${BOX}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== fc2 dg0.002 fd (Fermi-Dirac, T_el=316K) box $BOX start $(date) ==="
PW=/root/gpupw.sh; PD=/root/phonon/pseudo
case "$BOX" in
  A) JOBS="NbSe2:3:results/v100/fc2_nbse2_3x3_0.002_fd NbS2:3:results/v100/fc2_nbs2_3x3_0.002_fd" ;;
  B) JOBS="2H-TaSe2:3:results/v100/fc2_tase2_3x3_0.002_fd 1T-VSe2:4:results/v100/fc2_vse2_4x4_0.002_fd" ;;
  *) echo "bad box $BOX"; exit 1 ;;
esac
for J in $JOBS; do
  IFS=':' read -r MAT SC WD <<< "$J"
  echo "===== $MAT sc=${SC}x dg0.002 fd $(date) ====="
  "$PY" scripts/v100/tmd_dft_fc2.py --name "$MAT" --pw "$PW" --nproc 1 \
       --pseudo-dir "$PD" --degauss 0.002 --smearing fd --supercell "$SC" --workdir "$WD" \
       || echo "!! $MAT FAILED"
done
echo "=== fc2 fd box $BOX DONE $(date) ==="
touch "results/v100/fc2_fd/DONE_${BOX}"
