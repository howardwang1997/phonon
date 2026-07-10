#!/usr/bin/env bash
# Monitor the 60-snapshot graphene DFT-MD (Box A) for the v4 distillation data.
# Emits when snaps_T300.npz is ready (the distillation input), + failures.
BOXA=root@100.80.236.112
SEENF=/tmp/seen_v4.txt; touch "$SEENF"
sq(){ ssh -o ControlPath=none -o ConnectTimeout=25 -o ServerAliveInterval=5 "$@" 2>/dev/null; }
seen(){ grep -qxF "$1" "$SEENF" 2>/dev/null; }
mark(){ echo "$1" >> "$SEENF"; }
hb=0
while true; do
  if ! seen "snaps"; then
    if sq "$BOXA" 'test -f /data/gr_dftmd60/T300/snaps_T300.npz'; then
      mark "snaps"; echo "GRAPHENE DFT-MD 60-snap DONE (Box A) -> snaps_T300.npz ready -> distill v4"
      break
    fi
    while IFS= read -r fl; do [ -z "$fl" ] && continue; seen "f:$fl" && continue; mark "f:$fl"; echo "DFT-MD issue: $fl"; done < <(sq "$BOXA" 'grep -E "Error|Traceback|FAILED|abort" /tmp/DMD.log 2>/dev/null | tail -2')
  fi
  hb=$((hb+1))
  if [ $((hb % 4)) -eq 0 ]; then
    last=$(sq "$BOXA" 'tail -1 /tmp/DMD.log 2>/dev/null | cut -c1-50')
    echo "HEARTBEAT: DMD-60 [$last]"
  fi
  sleep 900
done
