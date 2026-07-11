#!/usr/bin/env bash
# Monitor Box B 100K DFT-MD completion -> v5 distill (combine 100K + 300K data).
BOXB=root@100.123.220.57
SEENF=/tmp/seen_v5.txt; touch "$SEENF"
sq(){ ssh -o ControlPath=none -o ConnectTimeout=25 -o ServerAliveInterval=5 "$@" 2>/dev/null; }
seen(){ grep -qxF "$1" "$SEENF" 2>/dev/null; }
mark(){ echo "$1" >> "$SEENF"; }
hb=0
while true; do
  if ! seen "done"; then
    if sq "$BOXB" 'test -f /data/gr_dftmd100/T100/snaps_T100.npz'; then
      mark "done"; echo "BOX B 100K DFT-MD DONE -> snaps_T100.npz ready -> combine + distill v5"; break
    fi
    while IFS= read -r fl; do [ -z "$fl" ] && continue; seen "f:$fl" && continue; mark "f:$fl"; echo "Box B issue: $fl"; done < <(sq "$BOXB" 'grep -E "Error|Traceback|FAILED" /tmp/DMD100.log 2>/dev/null | tail -2')
  fi
  hb=$((hb+1))
  if [ $((hb % 4)) -eq 0 ]; then
    last=$(sq "$BOXB" 'tail -1 /tmp/DMD100.log 2>/dev/null | cut -c1-50')
    echo "HEARTBEAT: DMD100 [$last]"
  fi
  sleep 900
done
