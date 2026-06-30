#!/usr/bin/env bash
# At-a-glance V100 campaign status (run on either box).
#   bash scripts/v100/status.sh
set -uo pipefail
cd "$HOME/phonon"
echo "================ V100 campaign status $(date) ================"
echo "--- tmux (lanes + any running EPW campaign) ---"
tmux ls 2>/dev/null | grep -E "v100gpu|v100cpu|^j0|jconv|jlq6" || echo "(no campaign tmux)"

echo; echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
echo "load: $(uptime | sed 's/.*load average/load average/')"

echo; echo "--- GPU lane: DFT fc2 / bands / Path-P done ---"
echo "  fc2  : $(ls results/v100/fc2/*_phonopy.yaml 2>/dev/null | wc -l | tr -d ' ') materials"
ls results/v100/fc2/*_phonopy.yaml 2>/dev/null | sed 's#.*/##;s/_phonopy.yaml//' | tr '\n' ' '; echo
echo "  bands: $(ls results/v100/bands/*_ebands.npz 2>/dev/null | wc -l | tr -d ' ') materials"
echo "  pathP: $(ls data/v100/path_p/*/train.xyz 2>/dev/null | wc -l | tr -d ' ') materials"

echo; echo "--- CPU lane: (E)-channel EPW results ---"
cat /root/tmd_epw_results.csv 2>/dev/null || echo "(none yet)"

echo; echo "--- recent lane log tails ---"
for f in results/v100/lanelogs/gpu_*.log results/v100/lanelogs/cpu_*.log; do
  [ -f "$f" ] && { echo "### $(basename "$f")"; tail -n 3 "$f"; }
done
echo; echo "tip: full logs in results/v100/lanelogs/ ; summary = results/v100/SUMMARY.md (after aggregate.py)"
