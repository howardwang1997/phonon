#!/usr/bin/env bash
# =============================================================================
# Autonomous research loop. Waits for the current SSCHA follow-up + its finalize
# to drain, then runs the next conceived experiment (κ-vs-breadth), then commits
# & pushes. Extensible: append phases below.
#   nohup bash scripts/h20/research_loop.sh > results/h20/research_loop.out 2>&1 & disown
# =============================================================================
set -uo pipefail
cd "$HOME/phonon"
PY="$HOME/miniconda3/envs/phonon/bin/python"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a results/h20/research_loop.log; }

log "=== research loop start ==="

# Phase 0: wait for the SSCHA follow-up + its finalize to finish
log "phase 0: waiting for SSCHA follow-up + finalize to drain ..."
while pgrep -f 'run_sscha_followup.sh|finalize_and_push.sh' >/dev/null 2>&1; do sleep 60; done
log "phase 0 drained."

# Phase 1: κ-vs-distillation-breadth (NEW experiment)
log "phase 1: κ-vs-breadth (4 models x 5 materials, sc3) ..."
bash scripts/h20/run_kappa_breadth.sh >> results/h20/kappa_breadth.log 2>&1
log "phase 1 runs done. regenerating SUMMARY + commit/push"
"$PY" scripts/h20/aggregate.py >> results/h20/aggregate_final.log 2>&1
bash scripts/h20/finalize_and_push.sh >> results/h20/finalize.log 2>&1
log "phase 1 committed/pushed."

# --- append further conceived phases below this line ---

log "=== research loop end ==="
