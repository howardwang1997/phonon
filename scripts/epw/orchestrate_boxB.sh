#!/usr/bin/env bash
# Box B parallel orchestrator. Phase 1: NbSe2 degauss 0.05 ∥ 0.06 (concurrent tmux).
# Phase 2 (after phase 1): graphene lambda(nkf) conv ∥ NbSe2 lambda_q 6x6 (concurrent).
# ~10-16 cores in use. Per-point result files merged at the end. Runs in tmux 'qboxB'.
set -u
S=/root/phonon/scripts/epw
# --- Phase 1: degauss sweep (Box B share) ---
tmux new-session -d -s j050 "env DEGAUSS=0.05 NQ=3 NKF=24 WORK=/data/nbse2_d050 RESULT_CSV=/root/res_d050.csv bash $S/nbse2_epw_full.sh > /root/j050.log 2>&1"
tmux new-session -d -s j060 "env DEGAUSS=0.06 NQ=3 NKF=24 WORK=/data/nbse2_d060 RESULT_CSV=/root/res_d060.csv bash $S/nbse2_epw_full.sh > /root/j060.log 2>&1"
echo "### phase1 j050 ∥ j060 at $(date +%H:%M); waiting ..."
while tmux has-session -t j050 2>/dev/null || tmux has-session -t j060 2>/dev/null; do sleep 300; done
# --- Phase 2: graphene conv ∥ NbSe2 lambda_q 6x6 ---
tmux new-session -d -s jconv "bash $S/graphene_lambda_conv.sh > /root/gr_conv.log 2>&1"
tmux new-session -d -s jlq6 "env DEGAUSS=0.03 NQ=6 NKF=24 WORK=/data/nbse2_d030_q6 RESULT_CSV=/root/res_q6.csv bash $S/nbse2_epw_full.sh > /root/jlq6.log 2>&1"
echo "### phase2 jconv ∥ jlq6 at $(date +%H:%M); waiting ..."
while tmux has-session -t jconv 2>/dev/null || tmux has-session -t jlq6 2>/dev/null; do sleep 300; done
echo "degauss,T_el,E_F,minfreq_cm,lambda,lambda_tr,maxgamma_meV,NQ,NKF" > /root/nbse2_lambda_results_B.csv
cat /root/res_d050.csv /root/res_d060.csv /root/res_q6.csv 2>/dev/null >> /root/nbse2_lambda_results_B.csv
echo "=== Box B NbSe2 lambda ==="; cat /root/nbse2_lambda_results_B.csv
echo "QUEUE_BOXB_DONE"
