#!/usr/bin/env bash
# Durable watchdog for a single V100 recovery lane.
#
# It never kills a live calculation. If a calculation exists but has stopped
# updating its progress files, it records a stale warning for inspection. It
# only starts the systemd-owned recovery queue when the lane is incomplete and
# no relevant process is alive.
set -euo pipefail

BOX="${1:?usage: bash scripts/v100/recovery_watchdog.sh A|B}"
REPO_ROOT="${PHONON_REPO_ROOT:-/root/phonon}"
STALE_SECONDS="${PHONON_STALE_SECONDS:-3600}"

case "$BOX" in
  A)
    PROGRESS_FILES=(
      /data/gr_dftmd_recovery/A/T300/md_checkpoint/state.npz
      /data/gr_dftmd_recovery/A/T300/espresso.pwo
      /data/phonon_offload/recovery_queue_A.log
    )
    PROCESS_PATTERN='[r]un_post_tmd_recovery.sh A|[r]un_graphene_failed_points_recovery.sh A|[r]un_dftmd_recovery.sh A|[d]ft_md_tdep.py.*--workroot /data/gr_dftmd_recovery/A'
    ;;
  B)
    PROGRESS_FILES=(
      /data/v100scratch/graphene_recovery/graphene_sc6_dg0.20/disp-000/espresso.pwo
      /data/v100scratch/graphene_recovery/graphene_sc6_dg0.20/disp-001/espresso.pwo
      /data/v100scratch/graphene_recovery/graphene_sc6_dg0.20/disp-002/espresso.pwo
      /data/v100scratch/graphene_recovery/graphene_sc6_dg0.20/disp-003/espresso.pwo
      /data/gr_dftmd_recovery/B/T600/md_checkpoint/state.npz
      /data/gr_dftmd_recovery/B/T600/espresso.pwo
      /data/phonon_offload/recovery_queue_B.log
    )
    PROCESS_PATTERN='[r]un_post_tmd_recovery.sh B|[r]un_graphene_failed_points_recovery.sh B|[r]un_dftmd_recovery.sh B|[m]1_1b_graphene_dft.py.*graphene_sc6_dg0.20|[d]ft_md_tdep.py.*--workroot /data/gr_dftmd_recovery/B'
    ;;
  *)
    echo "unknown box '$BOX' (expected A or B)" >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"

DONE="/data/gr_dftmd_recovery/${BOX}/DONE"
PAUSE="/data/phonon_offload/recovery_pause_${BOX}"
RUN_UNIT="phonon-recovery-run@${BOX}.service"
STATUS="/data/phonon_offload/recovery_watch_${BOX}.status"
LOCK="/run/lock/phonon-recovery-watch-${BOX}.lock"

mkdir -p "$(dirname "$STATUS")"
exec 9>"$LOCK"
flock -n 9 || exit 0

record_status() {
  local state="$1" detail="$2" tmp="${STATUS}.tmp.$$"
  printf 'checked_at=%s\nbox=%s\nstate=%s\ndetail=%s\n' \
    "$(date -Is)" "$BOX" "$state" "$detail" >"$tmp"
  mv -f "$tmp" "$STATUS"
}

latest_progress_mtime() {
  local latest=0 file mtime
  for file in "${PROGRESS_FILES[@]}"; do
    [[ -e "$file" ]] || continue
    mtime="$(stat -c %Y -- "$file" 2>/dev/null || true)"
    [[ "$mtime" =~ ^[0-9]+$ ]] || continue
    if (( mtime > latest )); then
      latest="$mtime"
    fi
  done
  printf '%s\n' "$latest"
}

record_live_state() {
  local owner="$1" latest now age
  latest="$(latest_progress_mtime)"
  now="$(date +%s)"
  if (( latest > 0 )); then
    age=$((now - latest))
    if (( age >= STALE_SECONDS )); then
      record_status stale "${owner}; newest progress is ${age}s old; live process was not killed"
      echo "[$BOX] WARNING: live $owner process exists but progress is ${age}s old"
      return
    fi
    record_status running "${owner}; newest progress is ${age}s old"
  else
    record_status running "${owner}; no progress file exists yet"
  fi
}

if [[ -e "$DONE" ]]; then
  record_status complete "$DONE exists"
  exit 0
fi

# An explicit pause marker lets a higher-priority, checkpoint-safe job borrow
# the GPU without the timer immediately relaunching DFT-MD.  The borrowing job
# owns removal of this marker and then starts RUN_UNIT to resume from state.npz.
if [[ -e "$PAUSE" ]]; then
  record_status paused "$PAUSE exists; automatic restart intentionally suppressed"
  exit 0
fi

unit_state="$(systemctl is-active "$RUN_UNIT" 2>/dev/null || true)"
case "$unit_state" in
  active|activating|reloading)
    record_live_state systemd
    exit 0
    ;;
esac

# A generic pw.x guard is deliberately conservative: if another QE job owns
# the GPU, wait rather than risk launching a duplicate or competing workload.
if pgrep -f "$PROCESS_PATTERN" >/dev/null || pgrep -x pw.x >/dev/null; then
  record_live_state external
  exit 0
fi

record_status starting "no live recovery/QE process; requesting $RUN_UNIT"
echo "[$BOX] incomplete lane has no live process; starting $RUN_UNIT"
if systemctl start "$RUN_UNIT"; then
  record_status started "$RUN_UNIT accepted by systemd"
else
  record_status start_failed "systemctl could not start $RUN_UNIT; inspect journal/start limits"
  exit 1
fi
