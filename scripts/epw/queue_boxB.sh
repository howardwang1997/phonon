#!/usr/bin/env bash
# Box B automated queue. Phase 1 (dual-machine share): NbSe2 lambda(T_el) at
# degauss 0.05, 0.06. Phase 2 (box-B only): graphene lambda(nkf) convergence,
# then NbSe2 lambda_q at coarse 6x6. Each sub-job is fault-isolated (continue on
# failure). Results -> per-job CSVs. Run inside tmux 'qboxB'.
set -u
S=/root/phonon/scripts/epw
RES=/root/nbse2_lambda_results_B.csv
echo "degauss,T_el,E_F,minfreq_cm,lambda,lambda_tr,maxgamma_meV,NQ,NKF" > "$RES"
run() { echo "### $(date +%H:%M) START: $*"; ( "$@" ) || echo "### SUBJOB FAILED: $*"; echo "### $(date +%H:%M) END: $*"; }

# --- Phase 1: NbSe2 lambda(T_el) (Box B share) ---
run env DEGAUSS=0.05 NQ=3 NKF=24 WORK=/data/nbse2_d050 RESULT_CSV=$RES bash $S/nbse2_epw_full.sh
run env DEGAUSS=0.06 NQ=3 NKF=24 WORK=/data/nbse2_d060 RESULT_CSV=$RES bash $S/nbse2_epw_full.sh

# --- Phase 2: graphene lambda(nkf) convergence ---
run bash $S/graphene_lambda_conv.sh

# --- Phase 3: NbSe2 lambda_q at coarse 6x6 (heavy) ---
run env DEGAUSS=0.03 NQ=6 NKF=24 WORK=/data/nbse2_d030_q6 RESULT_CSV=$RES bash $S/nbse2_epw_full.sh

echo "=== Box B results (NbSe2 lambda) ==="; cat "$RES" 2>/dev/null
echo "QUEUE_BOXB_DONE"
