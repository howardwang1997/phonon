#!/usr/bin/env bash
# Monitor graphene re-distill (2060) + DFT-MD-TDEP (Box A). Emits on completion + failures.
BOXA=root@100.80.236.112; R2060=howardwang@100.105.21.7
SEENF=/tmp/seen_graphene.txt; touch "$SEENF"; hb=0
sq(){ ssh -o ControlPath=none -o ConnectTimeout=25 -o ServerAliveInterval=5 "$@" 2>/dev/null; }
seen(){ grep -qxF "$1" "$SEENF" 2>/dev/null; }
mark(){ echo "$1" >> "$SEENF"; }
while true; do
  # 2060 re-distill done
  if ! seen "grbb"; then
    if sq "$R2060" 'test -f ~/phonon/results/gr_backbone_v2/.done'; then mark "grbb"; echo "GRAPHENE BACKBONE v2 DONE (2060) -> ready to re-deploy"; fi
    while IFS= read -r fl; do [ -z "$fl" ] && continue; seen "grbbf:$fl" && continue; mark "grbbf:$fl"; echo "2060 finetune issue: $fl"; done < <(sq "$R2060" 'grep -E "Error|Traceback|FAILED" /tmp/GR2060.log 2>/dev/null | tail -2')
  fi
  # Box A DFT-MD-TDEP done
  if ! seen "dmd"; then
    if sq "$BOXA" 'test -f /root/phonon/results/td_phonon/graphene_dft_tdep.npz'; then mark "dmd"; echo "GRAPHENE DFT-MD-TDEP DONE (Box A) -> ready to plot (L)"; fi
    while IFS= read -r fl; do [ -z "$fl" ] && continue; seen "dmdf:$fl" && continue; mark "dmdf:$fl"; echo "Box A DFT-MD issue: $fl"; done < <(sq "$BOXA" 'grep -E "Error|Traceback|FAILED|abort" /tmp/DMD.log 2>/dev/null | tail -2')
    if ! seen "dmdprog"; then
      last=$(sq "$BOXA" 'tail -1 /tmp/DMD.log 2>/dev/null | cut -c1-60')
      [ -n "$last" ] && { mark "dmdprog"; echo "DFT-MD progress: $last"; }
    fi
  fi
  # both done -> exit
  seen "grbb" && seen "dmd" && { echo "BOTH GRAPHENE JOBS DONE"; break; }
  hb=$((hb+1))
  if [ $((hb % 3)) -eq 0 ]; then
    g=$(sq "$R2060" 'tail -1 /tmp/GR2060.log 2>/dev/null | cut -c1-45')
    d=$(sq "$BOXA" 'tail -1 /tmp/DMD.log 2>/dev/null | cut -c1-45')
    echo "HEARTBEAT: 2060-ft[$g] | DMD[$d]"
  fi
  sleep 600
done
