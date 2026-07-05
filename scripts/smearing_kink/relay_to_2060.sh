#!/usr/bin/env bash
# Relay watcher: push any new data/path_p_<mat>/train.xyz from box -> 2060.
# Idempotent via .pushed marker. Runs until killed.
cd /root/phonon
while true; do
  for D in data/path_p_*; do
    [ -d "$D" ] || continue
    [ -f "$D/train.xyz" ] || continue
    [ -f "$D/.pushed" ] && continue
    scp -o ControlPath=none -q -r "$D" howardwang@100.105.21.7:~/phonon/data/ 2>/dev/null \
      && { touch "$D/.pushed"; echo "[relay] $(basename $D) -> 2060 $(date +%T)"; }
  done
  sleep 60
done
