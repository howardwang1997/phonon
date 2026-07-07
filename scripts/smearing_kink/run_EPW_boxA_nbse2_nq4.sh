#!/usr/bin/env bash
# NbSe2 EPW at NQ=4 (incommensurate with the 3x3 CDW q=1/3 -> soft mode NOT sampled ->
# clean, non-divergent lambda). Vets the lambda=17 soft-mode divergence found at NQ=3.
# Queued on Box A: WAITS for A2 (NbS2 4x4) to finish, then runs dg0.03 + dg0.04.
set -u
cd /root/phonon
echo "[epwNQ4] $(date) waiting for A2 (NbS2 4x4) to finish ..."
while tmux has-session -t A2 2>/dev/null; do sleep 120; done
echo "[epwNQ4] $(date) A2 done -> NbSe2 EPW NQ=4 (clean lambda)"
S=/root/phonon/scripts/epw
RES=/root/nbse2_lambda_nq4.csv
echo "degauss,T_el,E_F,minfreq_cm,lambda,lambda_tr,maxgamma_meV,NQ,NKF" > "$RES"
run(){ echo "### $(date +%H:%M) START d$1 NQ4"; ( "$@" ) || echo "### FAILED d$1"; }
run env DEGAUSS=0.03 NQ=4 NKF=24 WORK=/data/nbse2_d030_nq4 RESULT_CSV=$RES bash $S/nbse2_epw_full.sh
run env DEGAUSS=0.04 NQ=4 NKF=24 WORK=/data/nbse2_d040_nq4 RESULT_CSV=$RES bash $S/nbse2_epw_full.sh
echo "[epwNQ4] DONE $(date)"; echo "=== NbSe2 lambda NQ=4 (clean) ==="; cat "$RES"
