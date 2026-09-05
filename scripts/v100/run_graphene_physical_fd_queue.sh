#!/usr/bin/env bash
# Remote queue owner for the physical-FD DFPT campaigns.  This keeps the next
# campaign moving even if the Mac-side monitor is asleep; every compute runner
# is independently restartable and reuses completed SCF/DFPT points.
set -euo pipefail

LANE="${1:?usage: run_graphene_physical_fd_queue.sh A|B}"
case "$LANE" in
  A) CAMPAIGNS=(FD300_K144 FD300_LINE) ;;
  B) CAMPAIGNS=(FD600_LINE FD0_SCAN) ;;
  *) echo "unknown lane: $LANE" >&2; exit 2 ;;
esac

BASE=/data/graphene_physical_fd_dfpt/campaigns
for campaign in "${CAMPAIGNS[@]}"; do
    unit="phonon-graphene-physical-fd@${campaign}.service"
    while [[ ! -e "$BASE/$campaign/DONE" ]]; do
        state="$(systemctl is-active "$unit" 2>/dev/null || true)"
        if [[ "$state" != active && "$state" != activating ]]; then
            echo "[$(date -Is)] starting $unit (previous state: ${state:-unknown})"
            systemctl reset-failed "$unit" 2>/dev/null || true
            systemctl restart "$unit"
        fi
        sleep 60
    done
    echo "[$(date -Is)] $campaign complete"
done
echo "[$(date -Is)] physical-FD queue lane $LANE complete"
