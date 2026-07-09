#!/usr/bin/env bash
# Milestone monitor for the M1->M2/M4 + NbSe2-NQ4 campaign. Emits ONLY on key events.
# Polls every ~18 min; hourly heartbeat. Auto-launches are handled by the wait-drivers
# (run_M2_boxB_family_epw.sh waits for M1 tmux; run_M4_2060.sh waits for .M1_2060_all_done).
SS="-o ControlPath=none -o ConnectTimeout=10"
BOXA=root@100.80.236.112; BOXB=root@100.123.220.57; R2060=howardwang@100.105.21.7
declare -A SEEN; hb=0
while true; do
  # M1 2060 SSCHA results (new *_4x4.csv)
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    bn=$(basename "$f")
    [ -n "${SEEN[m1:$bn]:-}" ] && continue
    SEEN[m1:$bn]=1
    res=$(ssh $SS $R2060 "tail -4 ~/phonon/results/td_phonon/$bn 2>/dev/null | tr '\n' ' '" 2>/dev/null)
    echo "M1 SSCHA done ($bn): $res"
  done < <(ssh $SS $R2060 'ls ~/phonon/results/td_phonon/*_4x4.csv 2>/dev/null' 2>/dev/null)

  # M4 done / failed
  if [ -z "${SEEN[m4]:-}" ]; then
    if ssh $SS $R2060 'test -f ~/phonon/results/td_phonon/.M4_done' 2>/dev/null; then
      SEEN[m4]=1; echo "M4 DONE: B2d 2D kink(T_el,T_lat) surface complete on 2060"
    elif ssh $SS $R2060 'test -f ~/phonon/results/td_phonon/.M4_failed' 2>/dev/null; then
      SEEN[m4]=1; echo "M4 FAILED: smoke-test produced no csv (check /tmp/M4_wait.log on 2060)"
    fi
  fi

  # M2 lambda rows (new)
  while IFS= read -r r; do
    [ -z "$r" ] && continue; case "$r" in material,*) continue;; esac
    [ -n "${SEEN[m2:$r]:-}" ] && continue
    SEEN[m2:$r]=1; echo "M2 lambda: $r"
  done < <(ssh $SS $BOXB 'cat /root/family_lambda_stage2.csv 2>/dev/null' 2>/dev/null)

  # M1 Box B all done (triggers M2)
  if [ -z "${SEEN[m1b]:-}" ] && ! ssh $SS $BOXB 'tmux has-session -t M1 2>/dev/null' 2>/dev/null; then
    SEEN[m1b]=1; echo "M1 Box B DONE: 2H 4x4 all relayed (M2 wait-driver should now start EPW)"
  fi

  # NbSe2 NQ4 done
  if [ -z "${SEEN[nq4]:-}" ]; then
    nrows=$(ssh $SS $BOXA 'wc -l < /root/nbse2_lambda_nq4.csv 2>/dev/null' 2>/dev/null)
    if [ "${nrows:-0}" -gt 1 ] 2>/dev/null; then
      SEEN[nq4]=1; last=$(ssh $SS $BOXA 'tail -1 /root/nbse2_lambda_nq4.csv' 2>/dev/null); echo "NbSe2 NQ4 DONE: $last"
    fi
  fi

  # hourly heartbeat
  hb=$((hb+1))
  if [ $((hb % 4)) -eq 0 ]; then
    m1b=$(ssh $SS $BOXB 'tail -1 /tmp/M1_boxB.log 2>/dev/null | cut -c1-60' 2>/dev/null)
    m12060=$(ssh $SS $R2060 'tail -1 /tmp/M1_2060.log 2>/dev/null | cut -c1-60' 2>/dev/null)
    echo "HEARTBEAT: M1-boxB[$m1b] | M1-2060[$m12060]"
  fi
  sleep 1080
done
