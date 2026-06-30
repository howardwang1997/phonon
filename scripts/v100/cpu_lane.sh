#!/usr/bin/env bash
# CPU lane for one box: the (E)-channel — sequentially run DFPT-with-smearing +
# EPW gamma_qv for this box's assigned known-CDW materials. Waits for any
# currently-running EPW campaign (the lambda(T_el) tmux sessions) to free the CPU
# first, so it doesn't oversubscribe the 16 cores. Idempotent via tmd_epw_full's
# step guards + the per-material EPW_DONE marker.
#   bash scripts/v100/cpu_lane.sh A
set -uo pipefail
cd "$HOME/phonon"
BOX="${1:?need box A|B}"
CONDA="$HOME/miniconda3"
PY="$CONDA/envs/phonon/bin/python"
RESULT_CSV="/root/tmd_epw_results.csv"
[ -f "$RESULT_CSV" ] || echo "material,degauss,T_el,E_F,minfreq_cm,lambda,lambda_tr,maxgamma_meV,NQ,NKF" > "$RESULT_CSV"
PD=$("$PY" -c "import sys;sys.path.insert(0,'scripts/v100');import tmd_common as t;print(t.load_config()['runtime']['pseudo_dir'])")
NP=$("$PY" -c "import sys;sys.path.insert(0,'scripts/v100');import tmd_common as t;print(t.load_config()['runtime']['np_epw'])")
mkdir -p results/v100/lanelogs
LOG="results/v100/lanelogs/cpu_${BOX}.log"
echo "=== CPU lane box $BOX start $(date) ===" | tee "$LOG"

# 1) wait for any running lambda(T_el) campaign to free the CPU (don't oversubscribe)
echo "[cpu:$BOX] waiting for current EPW campaign tmux (j0*/jconv/jlq6) to clear ..." | tee -a "$LOG"
while tmux ls 2>/dev/null | grep -qE "^(j0[0-9]+|jconv|jlq6|grsweep):"; do sleep 300; done
echo "[cpu:$BOX] CPU free -> starting (E)-channel queue $(date)" | tee -a "$LOG"

# 2) run each assigned material through the full (E)-channel
"$PY" scripts/v100/tmd_common.py boxepw "$BOX" | while read -r MAT; do
  WORK="/data/${MAT}_epw"
  if [ -f "$WORK/EPW_DONE" ]; then echo "[cpu:$BOX] $MAT already done -> skip" | tee -a "$LOG"; continue; fi
  ENV=$("$PY" scripts/v100/tmd_common.py epwenv "$MAT")
  echo "===== [cpu:$BOX] (E)-channel $MAT $(date +%H:%M) =====" | tee -a "$LOG"
  env $ENV PSEUDO="$PD" WORK="$WORK" RESULT_CSV="$RESULT_CSV" NP="$NP" \
    bash scripts/v100/tmd_epw_full.sh >>"$LOG" 2>&1 \
    || echo "!! [cpu:$BOX] $MAT EPW FAILED (continuing)" | tee -a "$LOG"
done
echo "=== CPU lane box $BOX DONE $(date) ===" | tee -a "$LOG"
touch "results/v100/CPU_LANE_${BOX}_DONE"
