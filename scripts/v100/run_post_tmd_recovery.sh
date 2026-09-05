#!/usr/bin/env bash
# Continue each V100 lane in strict order after its TMD campaign succeeds:
# missing graphene point(s) -> checkpointed DFT-MD.  A failed/missing TMD
# marker stops the lane instead of allowing a false downstream completion.
set -euo pipefail

BOX="${1:?usage: bash scripts/v100/run_post_tmd_recovery.sh A|B}"
cd /root/phonon

case "$BOX" in
  A|B) ;;
  *)
    echo "unknown box '$BOX' (expected A or B)" >&2
    exit 2
    ;;
esac

# Only one recovery controller may own a lane. The watchdog also checks live
# compute processes before starting this script, while this lock closes the
# remaining race between two simultaneous launch attempts.
LOCK="/run/lock/phonon-recovery-lane-${BOX}.lock"
exec 8>"$LOCK"
if ! flock -n 8; then
  echo "[$BOX] another recovery controller already owns $LOCK; leaving it alone"
  exit 0
fi

TMD_MARKER="results/tmd_exp_a_recovery/DONE_${BOX}"
TMD_SESSION="recovery_tmd_${BOX}"
LOG="/data/phonon_offload/recovery_queue_${BOX}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== [$BOX] post-TMD recovery waiter start $(date -Is) ==="
while [[ ! -e "$TMD_MARKER" ]]; do
  if ! tmux has-session -t "$TMD_SESSION" 2>/dev/null; then
    echo "[$BOX] ERROR: $TMD_SESSION ended without validated marker $TMD_MARKER" >&2
    exit 1
  fi
  echo "[$BOX] waiting for validated TMD marker $(date -Is)"
  sleep 60
done

echo "[$BOX] TMD artifacts validated; starting graphene recovery $(date -Is)"
bash scripts/v100/run_graphene_failed_points_recovery.sh "$BOX"

echo "[$BOX] graphene artifacts validated; starting checkpointed DFT-MD $(date -Is)"
bash scripts/v100/run_dftmd_recovery.sh "$BOX"

echo "=== [$BOX] full recovery queue COMPLETE $(date -Is) ==="
