#!/usr/bin/env bash
# graphene lambda(n) sweep: run the doped-graphene EPW pipeline at several tot_charge
# values -> extract lambda vs doping. Turns the single E2b point (lambda=0.96 @ -0.04)
# into a lambda(n) trend line (the e-ph coupling grows as E_F moves into the band).
# Each doping has its own WORK dir + DFT (screening changes with doping). 24x24 fine
# grid for speed. Run in tmux on box A (qe env).
set -u
DRIVER=/root/phonon/scripts/epw/run_graphene_doped_epw.sh
CSV=/root/gr_lambda_sweep.csv
echo "tot_charge,E_F_eV,lambda,lambda_tr" > "$CSV"
for DOP in -0.01 -0.02 -0.04 -0.08; do
  tag=$(echo "$DOP" | tr -cd '0-9')
  W=/data/gr_dope_$tag
  echo "===== [sweep] doping tot_charge=$DOP  (WORK=$W) ====="
  if WORK="$W" TOT_CHARGE="$DOP" NKF=24 NQF=24 bash "$DRIVER"; then
    EF=$(grep -i 'the Fermi energy' "$W/scf.out" 2>/dev/null | tail -1 | grep -oE '[-0-9.]+' | head -1)
    LAM=$(grep -aE "lambda :" "$W/epw.out" 2>/dev/null | tail -1 | grep -oE "[0-9.]+" | head -1)
    LAMTR=$(grep -aE "lambda_tr :" "$W/epw.out" 2>/dev/null | tail -1 | grep -oE "[0-9.]+" | head -1)
    echo "$DOP,${EF:-NA},${LAM:-NA},${LAMTR:-NA}" | tee -a "$CSV"
  else
    echo "$DOP,FAILED,NA,NA" | tee -a "$CSV"
  fi
done
echo "=== lambda(n) sweep result ==="; cat "$CSV"
echo "GR_LAMBDA_SWEEP_DONE"
