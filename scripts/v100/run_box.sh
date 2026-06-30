#!/usr/bin/env bash
# Per-box master for the V100 FP64 campaign. Launches BOTH lanes in tmux:
#   GPU lane  -> starts NOW (fills the idle GPU with fc2/bands/Path-P)
#   CPU lane  -> (E)-channel EPW; waits for any running lambda(T_el) campaign first
# Idempotent/resumable: re-run to continue. Run this on EACH box with its letter.
#   bash scripts/v100/run_box.sh A     # on Box A
#   bash scripts/v100/run_box.sh B     # on Box B
set -uo pipefail
cd "$HOME/phonon"
BOX="${1:?usage: run_box.sh A|B}"
mkdir -p results/v100

echo "############ V100 box $BOX campaign $(date) ############"

# Phase 0: pseudopotentials (idempotent)
PD=$("$HOME/miniconda3/envs/phonon/bin/python" -c \
   "import sys;sys.path.insert(0,'scripts/v100');import tmd_common as t;print(t.load_config()['runtime']['pseudo_dir'])")
bash scripts/v100/fetch_pseudos.sh "$PD" || {
  echo "!! some pseudos missing — Ta/Ti/V/S materials will fail until relayed (see log). Continuing."; }

# GPU lane (now) + CPU lane (waits for current campaign) — each in its own tmux
tmux has-session -t "v100gpu$BOX" 2>/dev/null \
  && echo "v100gpu$BOX already running" \
  || tmux new-session -d -s "v100gpu$BOX" "bash scripts/v100/gpu_lane.sh $BOX"
tmux has-session -t "v100cpu$BOX" 2>/dev/null \
  && echo "v100cpu$BOX already running" \
  || tmux new-session -d -s "v100cpu$BOX" "bash scripts/v100/cpu_lane.sh $BOX"

echo "launched: tmux v100gpu$BOX (GPU lane, now) + v100cpu$BOX (CPU lane, after current campaign)"
echo "watch:  bash scripts/v100/status.sh"
echo "logs:   results/v100/lanelogs/{gpu,cpu}_${BOX}.log"
