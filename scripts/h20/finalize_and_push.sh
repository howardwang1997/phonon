#!/usr/bin/env bash
# =============================================================================
# Finalize: wait for the SSCHA follow-up to drain, regenerate SUMMARY, then
# commit + push everything (campaign results + the execution fixes). Autonomous.
#   nohup bash scripts/h20/finalize_and_push.sh > results/h20/finalize.log 2>&1 & disown
# =============================================================================
set -uo pipefail
cd "$HOME/phonon"
PY="$HOME/miniconda3/envs/phonon/bin/python"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "=== finalize: waiting for SSCHA follow-up to drain ==="
while pgrep -f 'run_sscha_followup.sh' >/dev/null 2>&1; do sleep 60; done
log "SSCHA follow-up finished."

# 1. regenerate SUMMARY with all data (convergence-guarded SSCHA + distilled κ)
log "regenerating SUMMARY.md ..."
"$PY" scripts/h20/aggregate.py > results/h20/aggregate_final.log 2>&1 \
  && log "aggregate OK" || log "aggregate FAILED (see aggregate_final.log)"

# 2. stage results + execution-fix code. .gitignore handles exclusions.
git add -A results scripts src configs docs 2>/dev/null
# never stage the canon .model weights or giant xyz even if not ignored
git reset -- 'results/ablation/*.model' 'data/*.xyz' '*.pth' 2>/dev/null || true

staged=$(git diff --cached --name-only | wc -l)
log "staged $staged files"
if [ "$staged" -eq 0 ]; then
  log "nothing to commit — done."
  exit 0
fi

# 3. commit
git config user.email "howardwang1997@gmail.com" 2>/dev/null
git config user.name  "howardwang1997" 2>/dev/null
if git commit -q -m "H20 campaign results + autonomous execution fixes

Campaign 132/132 (Wave A+B): E1 atlas (6 MLIPs x 109 MDR), E3/E4 distillation,
(L)-channel TMD origin-map, E9 kappa. Plus follow-ups:
- E9 kappa with distilled canon model: Si sc4=139.8 (exp 140, -0.1%), C +15%;
  Ge/GaAs/BAs underestimated (model fc3 limitation).
- (L)-channel SSCHA follow-up with improved convergence (more pops, smaller step).
- aggregate.py convergence guard: flags diverged SSCHA (L-...-diverged) instead
  of reporting non-physical min-freqs.

Execution fixes that unblocked the campaign on this 8xH20 box:
- queue.py -> hqueue.py (was shadowing stdlib queue, broke import torch)
- model envs: install seekpath (silent all-ERR E1 csvs)
- sscha14 build: --no-build-isolation + setuptools<66
- mattersim torch 2.12+cu130 -> 2.6.0+cu124
- mace-omat: pop 'model' kwarg before mace_mp
- MDR pool network blocked -> pool_size capped to 109 cached materials
- autonomous infra: auto_driver.sh (self-healing re-launch), watchdog.sh" ; then
  log "commit OK"
else
  log "commit FAILED (nothing changed?)"; exit 1
fi

# 4. integrate any upstream changes, then push
log "fetching upstream ..."
if git fetch origin td-phonon-anomaly >>results/h20/push.log 2>&1; then
  if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/td-phonon-anomaly)" ] && \
     ! git merge-base --is-ancestor origin/td-phonon-anomaly HEAD; then
    log "upstream moved -> rebase ..."
    git rebase origin/td-phonon-anomaly >>results/h20/push.log 2>&1 || log "rebase had conflict (push will fail)"
  fi
fi
log "pushing ..."
if git push origin HEAD 2>>results/h20/push.log; then
  log "PUSH OK"
else
  log "PUSH FAILED (see push.log) — commit is local."
fi
log "=== finalize done ==="
