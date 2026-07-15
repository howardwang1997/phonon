#!/usr/bin/env bash
# Graphene Kohn-anomaly (E)-channel DFT prerequisite: PBE-relax a, then FD fc2 at
# 10 Fermi-Dirac smearing points (0.01->0.20, T_el 1579->31577 K). Mirrors Piscanec
# 2004 (Kohn anomaly vs electronic parameter), smearing as the knob. Every smearing
# uses the SAME relaxed a. Run in tmux ON the box.
#   A: relax + sharp 5 (0.01 0.015 0.02 0.03 0.04)   B: relax + broad 5 (0.06 0.08 0.10 0.14 0.20)
set -uo pipefail
BOX="${1:?need A|B}"
cd "$HOME/phonon"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate phonon
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="$HOME/miniconda3/envs/phonon/bin/python"
mkdir -p results/graphene_kohn_fd
LOG="results/graphene_kohn_fd/run_${BOX}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== graphene Kohn fd (relax + 10pt scan) box $BOX start $(date) ==="
PW=/root/gpupw.sh; PD=/root/phonon/pseudo
echo "--- vc-relax graphene a (PBE, fd, 2Dxy) ---"
"$PY" scripts/v100/relax_graphene_a.py --pw "$PW" --pseudo-dir "$PD" \
     --workdir "results/graphene_kohn_fd/relax_${BOX}" 2>results/graphene_kohn_fd/relax_${BOX}.err
AREL=$(cat "results/graphene_kohn_fd/relax_${BOX}/a_relaxed.txt" 2>/dev/null | tr -d '[:space:]')
echo "relaxed a = ${AREL} A  (start 2.46)"
case "$BOX" in
  A) DGS="0.01 0.015 0.02 0.03 0.04" ;;
  B) DGS="0.06 0.08 0.10 0.14 0.20" ;;
  *) echo "bad box $BOX"; exit 1 ;;
esac
for dg in $DGS; do
  TEL=$(awk "BEGIN{printf \"%.0f\", $dg*157887}")
  echo "===== graphene sc6 a=${AREL} dg=$dg fd (T_el=${TEL}K) $(date) ====="
  "$PY" scripts/m1_1b_graphene_dft.py --pw "$PW" --nproc 1 --pseudo-dir "$PD" \
       --pseudo C_ONCV_PBE-1.2.upf --a "$AREL" --supercell 6 --ecutwfc 60 --ecutrho 240 \
       --kpts 6 --disp 0.03 --smearing fd --degauss "$dg" \
       --workdir results/graphene_kohn_fd --tag "graphene_sc6_dg${dg}" \
       || echo "!! dg=$dg FAILED"
done
echo "=== graphene Kohn fd box $BOX DONE $(date) ==="
touch "results/graphene_kohn_fd/DONE_${BOX}"
