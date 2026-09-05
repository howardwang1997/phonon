#!/usr/bin/env bash
set -euo pipefail
LANE="${1:?usage: run_graphene_fd_label_relay_wave3.sh A|B}"
case "$LANE" in A|B) ;; *) exit 2 ;; esac
SOURCE="/data/graphene_fd_thermal_labels_wave3/$LANE"
REMOTE="howardwang@100.105.21.7"
DEST="/home/howardwang/phonon/data/graphene_fd_thermal_labels"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
SCP=(scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)

while [[ ! -e "$SOURCE/DONE" && ! -e "$SOURCE/SKIPPED_GATE_PASSED" ]]; do sleep 60; done
if [[ -e "$SOURCE/SKIPPED_GATE_PASSED" ]]; then exit 0; fi
[[ -s "$SOURCE/summary.json" && -s "$SOURCE/summary.xyz" ]]
"${SSH[@]}" "$REMOTE" "mkdir -p '$DEST'"
"${SCP[@]}" "$SOURCE/summary.json" "$REMOTE:$DEST/${LANE}3.json.partial"
"${SCP[@]}" "$SOURCE/summary.xyz" "$REMOTE:$DEST/${LANE}3.xyz.partial"
"${SSH[@]}" "$REMOTE" \
    "mv '$DEST/${LANE}3.json.partial' '$DEST/${LANE}3.json'; \
     mv '$DEST/${LANE}3.xyz.partial' '$DEST/${LANE}3.xyz'; \
     if test -s '$DEST/A3.xyz' && test -s '$DEST/B3.xyz' && ! test -e '$DEST/.READY_WAVE3'; then \
       touch '$DEST/.READY_WAVE3'; \
       systemctl --user restart phonon-graphene-fd-thermal-refine-wave3.service; \
     fi"
touch "$SOURCE/RELAYED"
