#!/usr/bin/env bash
# Milestone monitor for the M1->M2/M4 + NbSe2-NQ4 campaign (bash 3.2 compatible).
# Emits ONLY on key events. Polls every ~18 min; hourly heartbeat.
# Auto-launches are handled by the wait-drivers (M2 waits for M1 tmux; M4 for .M1_2060_all_done).
BOXA=root@100.80.236.112; BOXB=root@100.123.220.57; R2060=howardwang@100.105.21.7
SEENF=/tmp/campaign_seen.txt; touch "$SEENF"; hb=0
sq(){ ssh -o ControlPath=none -o ConnectTimeout=10 "$@" 2>/dev/null; }
seen(){ grep -qxF "$1" "$SEENF" 2>/dev/null; }
mark(){ echo "$1" >> "$SEENF"; }

while true; do
  # M1 2060 SSCHA results (new *_4x4.csv)
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    bn=$(basename "$f")
    seen "m1:$bn" && continue
    mark "m1:$bn"
    res=$(sq "$R2060" "tail -4 ~/phonon/results/td_phonon/$bn 2>/dev/null | tr '\n' ' '")
    echo "M1 SSCHA done ($bn): $res"
  done < <(sq "$R2060" 'ls ~/phonon/results/td_phonon/*_4x4.csv 2>/dev/null')

  # M4 done / failed
  if ! seen "m4"; then
    if sq "$R2060" 'test -f ~/phonon/results/td_phonon/.M4_done'; then
      mark "m4"; echo "M4 DONE: B2d 2D kink(T_el,T_lat) surface complete on 2060"
    elif sq "$R2060" 'test -f ~/phonon/results/td_phonon/.M4_failed'; then
      mark "m4"; echo "M4 FAILED: smoke-test produced no csv (check /tmp/M4_wait.log on 2060)"
    fi
  fi

  # M2 lambda rows (new)
  while IFS= read -r r; do
    [ -z "$r" ] && continue
    case "$r" in material,*) continue;; esac
    seen "m2:$r" && continue
    mark "m2:$r"; echo "M2 lambda: $r"
  done < <(sq "$BOXB" 'cat /root/family_lambda_stage2.csv 2>/dev/null')

  # M1 Box B all done — positive check on the driver's completion echo (robust to SSH hiccups;
  # a transient ssh failure must NOT look like "done"). M2 wait-driver checks tmux locally on Box B.
  if ! seen "m1b"; then
    if sq "$BOXB" 'grep -q "BOX B 2H-4x4 ALL DONE" /tmp/M1_boxB.log 2>/dev/null'; then
      mark "m1b"; echo "M1 Box B DONE: 2H 4x4 all relayed (M2 wait-driver should now start EPW)"
    fi
  fi

  # NbSe2 NQ4 done
  if ! seen "nq4"; then
    nrows=$(sq "$BOXA" 'wc -l < /root/nbse2_lambda_nq4.csv 2>/dev/null')
    if [ "${nrows:-0}" -gt 1 ] 2>/dev/null; then
      mark "nq4"; last=$(sq "$BOXA" 'tail -1 /root/nbse2_lambda_nq4.csv')
      echo "NbSe2 NQ4 DONE: $last"
    fi
  fi

  # hourly heartbeat
  hb=$((hb+1))
  if [ $((hb % 4)) -eq 0 ]; then
    m1b=$(sq "$BOXB" 'tail -1 /tmp/M1_boxB.log 2>/dev/null | cut -c1-55')
    m12060=$(sq "$R2060" 'tail -1 /tmp/M1_2060.log 2>/dev/null | cut -c1-55')
    echo "HEARTBEAT: M1-boxB[$m1b] | M1-2060[$m12060]"
  fi
  sleep 1080
done
