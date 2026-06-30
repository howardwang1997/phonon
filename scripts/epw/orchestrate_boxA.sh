#!/usr/bin/env bash
# Box A parallel orchestrator: run both NbSe2 degauss points (0.03, 0.04) CONCURRENTLY
# in separate tmux (16 cores; one point in EPW ~2c + one in DFPT ~8c = ~10c, no
# oversubscription). DFPT of already-started points is reused via step-guards. Each
# point writes its own result file; merged at the end. Runs inside tmux 'qboxA'.
set -u
S=/root/phonon/scripts/epw
tmux new-session -d -s j030 "env DEGAUSS=0.03 NQ=3 NKF=24 WORK=/data/nbse2_d030 RESULT_CSV=/root/res_d030.csv bash $S/nbse2_epw_full.sh > /root/j030.log 2>&1"
tmux new-session -d -s j040 "env DEGAUSS=0.04 NQ=3 NKF=24 WORK=/data/nbse2_d040 RESULT_CSV=/root/res_d040.csv bash $S/nbse2_epw_full.sh > /root/j040.log 2>&1"
echo "### launched j030 ∥ j040 at $(date +%H:%M); waiting ..."
while tmux has-session -t j030 2>/dev/null || tmux has-session -t j040 2>/dev/null; do sleep 300; done
echo "degauss,T_el,E_F,minfreq_cm,lambda,lambda_tr,maxgamma_meV,NQ,NKF" > /root/nbse2_lambda_results_A.csv
cat /root/res_d030.csv /root/res_d040.csv 2>/dev/null >> /root/nbse2_lambda_results_A.csv
echo "=== Box A NbSe2 lambda(T_el) ==="; cat /root/nbse2_lambda_results_A.csv
echo "QUEUE_BOXA_DONE"
