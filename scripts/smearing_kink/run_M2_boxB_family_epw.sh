#!/usr/bin/env bash
# M2: Family EPW stage 2 — λ(T_el) for NbS2/TaS2/TaSe2/TiSe2 at incommensurate NQ
# (clean, non-divergent λ) = (E)-channel first-principles arbiter for the origin discovery.
# WAITS for M1 (Box B tmux) to finish, then runs 2 smearings/material. ~14h.
#
# NQ strategy (avoid sampling each material's CDW q -> no soft-mode divergence):
#   2H CDW q=1/3 (NbS2/TaS2/TaSe2) -> NQ=4 ;  1T TiSe2 q=1/2 (M) -> NQ=3.
# DZ (X internal z) = 0.0485 default (NbSe2 value); refine per-material before production.
set -u
cd /root/phonon
S=scripts/epw/tmd_epw_full.sh
RES=/root/family_lambda_stage2.csv
[ -f "$RES" ] || echo "material,polytype,formula,degauss,T_el,E_F,minfreq_cm,lambda,lambda_tr,maxgamma_meV,NQ,NKF" > "$RES"
AB2BOHR=1.8897259886; VAC=31.0; DZ=${DZ:-0.0485}

echo "[M2] $(date) waiting for M1 (Box B tmux) to finish..."
while tmux has-session -t M1 2>/dev/null; do sleep 300; done
echo "[M2] $(date) M1 done -> Family EPW stage 2"

run_mat(){ # MAT M_ELM X_ELM M_MASS X_MASS POLY AANG THK NQ
  local MAT=$1 M=$2 X=$3 MM=$4 XM=$5 P=$6 A=$7 T=$8 NQ=$9
  local ABOHR COA DG TEL
  ABOHR=$(awk "BEGIN{printf \"%.4f\", $A*$AB2BOHR}")
  COA=$(awk "BEGIN{printf \"%.4f\", ($T+$VAC)/$A}")
  echo "[M2] === $MAT ($P-${M}${X}) A=$A->celldm1=$ABOHR COA=$COA NQ=$NQ ==="
  for DG in 0.03 0.04; do
    if grep -q "^${MAT},.*,${DG}," "$RES" 2>/dev/null; then echo "[M2] $MAT d$DG done, skip"; continue; fi
    MAT=$MAT M_ELM=$M X_ELM=$X M_MASS=$MM X_MASS=$XM POLYTYPE=$P \
      ABOHR=$ABOHR COA=$COA DZ=$DZ DEGAUSS=$DG NQ=$NQ NKF=24 NQF=24 NBND=30 NBNDSUB=11 \
      RESULT_CSV=$RES WORK=/data/${MAT}_d${DG}_q${NQ} NP=8 bash $S || echo "[M2] $MAT d$DG FAILED"
  done
}

#                MAT    M  X  Mmass  Xmass  POLY A    Thk  NQ
run_mat  nbs2   Nb S  92.906 32.06  2H   3.33 3.00 4
run_mat  tas2   Ta S  180.95 32.06  2H   3.31 3.00 4
run_mat  tase2  Ta Se 180.95 78.971 2H   3.43 3.30 4
run_mat  tise2  Ti Se 47.867 78.971 1T   3.53 2.90 3
echo "[M2] FAMILY EPW STAGE 2 DONE $(date)"; echo "=== results ==="; cat "$RES"
