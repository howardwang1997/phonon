#!/usr/bin/env bash
# Graphene (E)-channel Kohn-kink vs electronic smearing, Fermi-Dirac.
# degauss = k_B*T_el EXACTLY (fd). Establishes the fd standard for graphene
# (replaces the cold-smearing vq_kink6 set). 6x6 supercell (K=(1/3,1/3) on grid),
# ecutwfc 60, kpts 6 (dense for the Kohn anomaly). 6 key smearings, 3 per box.
# Run in tmux ON the box.   A: 0.002 0.005 0.010   B: 0.020 0.040 0.080
set -uo pipefail
BOX="${1:?need A|B}"
cd "$HOME/phonon"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate phonon
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="$HOME/miniconda3/envs/phonon/bin/python"
mkdir -p results/vq_kink6_fd
LOG="results/vq_kink6_fd/run_${BOX}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== graphene fd (Fermi-Dirac) 6x6 scan box $BOX start $(date) ==="
PW=/root/gpupw.sh; PD=/root/phonon/pseudo
case "$BOX" in
  A) DGS="0.002 0.005 0.010" ;;
  B) DGS="0.020 0.040 0.080" ;;
  *) echo "bad box $BOX"; exit 1 ;;
esac
for dg in $DGS; do
  TEL=$(awk "BEGIN{printf \"%.0f\", $dg*157887}")
  echo "===== graphene sc6 dg=$dg fd (T_el=${TEL}K) $(date) ====="
  "$PY" scripts/m1_1b_graphene_dft.py --pw "$PW" --nproc 1 \
       --pseudo-dir "$PD" --pseudo C_ONCV_PBE-1.2.upf \
       --a 2.46 --supercell 6 --ecutwfc 60 --ecutrho 240 --kpts 6 --disp 0.03 \
       --smearing fd --degauss "$dg" \
       --workdir results/vq_kink6_fd --tag "graphene_sc6_dg${dg}" \
       || echo "!! dg=$dg FAILED"
done
echo "=== graphene fd box $BOX DONE $(date) ==="
touch "results/vq_kink6_fd/DONE_${BOX}"
