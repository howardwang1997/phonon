#!/usr/bin/env bash
# At-a-glance campaign progress (no SSH-from-author needed; run it on the box).
#   bash scripts/h20/status.sh
set -uo pipefail
cd "$HOME/phonon"
PHPY="$HOME/miniconda3/envs/phonon/bin/python"
H=results/h20

echo "================= H20 campaign status $(date) ================="
if [ -f "$H/jobs.jsonl" ]; then
  "$PHPY" scripts/h20/hqueue.py status
else
  echo "(no manifest yet — run scripts/h20/run_campaign.sh)"
fi

echo
echo "--- GPU utilisation ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
  || echo "(nvidia-smi unavailable)"

echo
echo "--- most recent job log lines ---"
ls -t "$H"/joblogs/*.log 2>/dev/null | head -4 | while read -r f; do
  echo "### $(basename "$f")"
  tail -n 3 "$f"
done

echo
echo "--- bootstrap readiness ---"
grep -E 'env .*(READY|NOT READY)|E1 pool:' "$H/BOOTSTRAP_STATUS.txt" 2>/dev/null || echo "(no bootstrap status yet)"

echo
echo "tip: full driver log = tail -f $H/campaign.log ; results summary = $H/SUMMARY.md"
