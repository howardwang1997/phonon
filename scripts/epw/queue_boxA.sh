#!/usr/bin/env bash
# Box A automated queue. Dual-machine share of the NbSe2 lambda(T_el) sweep:
# degauss 0.03, 0.04. First WAITS for the running graphene doping sweep (tmux
# 'grsweep') to finish so it doesn't contend. Fault-isolated. Run inside tmux 'qboxA'.
set -u
S=/root/phonon/scripts/epw
RES=/root/nbse2_lambda_results_A.csv
echo "### $(date +%H:%M) waiting for grsweep to finish ..."
while tmux has-session -t grsweep 2>/dev/null; do sleep 300; done
echo "### $(date +%H:%M) grsweep done -> starting NbSe2 lambda(T_el) Box A share"
echo "degauss,T_el,E_F,minfreq_cm,lambda,lambda_tr,maxgamma_meV,NQ,NKF" > "$RES"
run() { echo "### $(date +%H:%M) START: $*"; ( "$@" ) || echo "### SUBJOB FAILED: $*"; echo "### $(date +%H:%M) END: $*"; }

run env DEGAUSS=0.03 NQ=3 NKF=24 WORK=/data/nbse2_d030 RESULT_CSV=$RES bash $S/nbse2_epw_full.sh
run env DEGAUSS=0.04 NQ=3 NKF=24 WORK=/data/nbse2_d040 RESULT_CSV=$RES bash $S/nbse2_epw_full.sh

echo "=== Box A results (NbSe2 lambda) ==="; cat "$RES" 2>/dev/null
echo "QUEUE_BOXA_DONE"
