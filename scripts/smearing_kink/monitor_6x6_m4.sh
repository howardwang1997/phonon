#!/usr/bin/env bash
# Lean monitor for A2-clean 6×6 fc2 (NbS2 BoxA, TaSe2 BoxB) + M4 B2d 2D (2060).
# Emits on completion + failures; hourly heartbeat. (bash 3.2, sq wrapper, 25s timeout.)
BOXA=root@100.80.236.112; BOXB=root@100.123.220.57; R2060=howardwang@100.105.21.7
SEENF=/tmp/seen_6x6m4.txt; touch "$SEENF"; hb=0
sq(){ ssh -o ControlPath=none -o ConnectTimeout=25 -o ServerAliveInterval=5 "$@" 2>/dev/null; }
seen(){ grep -qxF "$1" "$SEENF" 2>/dev/null; }
mark(){ echo "$1" >> "$SEENF"; }
while true; do
  # 6x6 fc2 done (.6x6_done marker) + bare min freq
  for spec in "$BOXA:NbS2:/tmp/A2_nbs2.log" "$BOXB:2H-TaSe2:/tmp/A2_tase2.log"; do
    H=${spec%%:*}; rest=${spec#*:}; M=${rest%%:*}; L=${rest##*:}
    if ! seen "6x6:$M"; then
      if sq "$H" "test -f /root/phonon/results/v100/fc2_${M}_6x6_0.015/.6x6_done"; then
        mf=$(sq "$H" "grep 'min freq' $L 2>/dev/null | tail -1")
        mark "6x6:$M"; echo "6x6 fc2 DONE ($M): $mf"
      fi
    fi
    # failures for this material
    while IFS= read -r fl; do
      [ -z "$fl" ] && continue; seen "f:$M:$fl" && continue; mark "f:$M:$fl"; echo "FAIL ($M): $fl"
    done < <(sq "$H" "grep -E 'FAIL|Traceback|Error:' $L 2>/dev/null | tail -2")
  done
  # M4 done / failed
  if ! seen "m4"; then
    if sq "$R2060" 'test -f ~/phonon/results/td_phonon/.M4_done'; then mark "m4"; echo "M4 DONE: b2d 2D kink(T_el,T_lat) csv ready on 2060 (replot locally — header bug)"; fi
    if sq "$R2060" 'test -f ~/phonon/results/td_phonon/.M4_failed'; then mark "m4"; echo "M4 FAILED (smoke-test)"; fi
  fi
  if ! seen "m4f2"; then
    while IFS= read -r fl; do [ -z "$fl" ] && continue; seen "m4f2:$fl" && continue; mark "m4f2:$fl"; echo "M4 issue: $fl"; done < <(sq "$R2060" 'grep -E "Traceback|Error" /tmp/M4_wait.log 2>/dev/null | tail -2')
  fi
  # hourly heartbeat
  hb=$((hb+1))
  if [ $((hb % 4)) -eq 0 ]; then
    a=$(sq "$BOXA" 'tail -1 /tmp/A2_nbs2.log 2>/dev/null|cut -c1-45')
    b=$(sq "$BOXB" 'tail -1 /tmp/A2_tase2.log 2>/dev/null|cut -c1-45')
    m=$(sq "$R2060" 'tail -1 /tmp/M4_wait.log 2>/dev/null|cut -c1-45')
    echo "HEARTBEAT: nbs2-6x6[$a] tase2-6x6[$b] M4[$m]"
  fi
  sleep 900
done
