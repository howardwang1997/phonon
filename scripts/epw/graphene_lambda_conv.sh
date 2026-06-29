#!/usr/bin/env bash
# graphene lambda convergence at fixed doping (tot_charge=-0.04): sweep the EPW fine
# grid nkf -> lambda(nkf). Reuses one DFT (scf+DFPT 6x6+nscf) via the doped driver's
# step-guards; only epw.x re-runs per nkf (epwwrite each time). Resolves the E2b
# "lambda=0.96 unconverged" question -> the plateau value is the trustworthy lambda.
set -u
DRIVER=/root/phonon/scripts/epw/run_graphene_doped_epw.sh
CSV=/root/gr_lambda_conv.csv
echo "nkf,E_F_eV,lambda,lambda_tr" > "$CSV"
for NKF in 36 60 90 120; do
  echo "===== [gr-conv] nkf=$NKF (tot_charge=-0.04) ====="
  if WORK=/data/gr_conv TOT_CHARGE=-0.04 NKF=$NKF NQF=$NKF bash "$DRIVER" > /root/gr_conv_$NKF.log 2>&1; then
    EF=$(grep -i 'the Fermi energy' /data/gr_conv/scf.out 2>/dev/null | tail -1 | grep -oE '[-0-9.]+' | head -1)
    LAM=$(grep -aE "lambda :" /data/gr_conv/epw.out 2>/dev/null | tail -1 | grep -oE "[-0-9.]+" | head -1)
    LAMTR=$(grep -aE "lambda_tr :" /data/gr_conv/epw.out 2>/dev/null | tail -1 | grep -oE "[-0-9.]+" | head -1)
    echo "$NKF,${EF:-NA},${LAM:-NA},${LAMTR:-NA}" | tee -a "$CSV"
  else
    echo "$NKF,FAILED,NA,NA" | tee -a "$CSV"
  fi
done
echo "=== graphene lambda(nkf) convergence ==="; cat "$CSV"
echo "GRAPHENE_CONV_DONE"
