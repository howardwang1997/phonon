#!/usr/bin/env bash
# Relay the second physical-FD label wave directly to the RTX 2060 over the
# Tailscale 100.x network.
set -euo pipefail

LANE="${1:?usage: run_graphene_fd_label_relay_wave2.sh A|B}"
case "$LANE" in A|B) ;; *) echo "unknown lane: $LANE" >&2; exit 2 ;; esac
SOURCE="/data/graphene_fd_thermal_labels_wave2/$LANE"
REMOTE="howardwang@100.105.21.7"
DEST="/home/howardwang/phonon/data/graphene_fd_thermal_labels"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
SCP=(scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)

while [[ ! -e "$SOURCE/DONE" ]]; do sleep 60; done
[[ -s "$SOURCE/summary.json" && -s "$SOURCE/summary.xyz" ]]
"${SSH[@]}" "$REMOTE" "mkdir -p '$DEST'"
"${SCP[@]}" "$SOURCE/summary.json" "$REMOTE:$DEST/${LANE}2.json.partial"
"${SCP[@]}" "$SOURCE/summary.xyz" "$REMOTE:$DEST/${LANE}2.xyz.partial"
"${SSH[@]}" "$REMOTE" \
    "mv '$DEST/${LANE}2.json.partial' '$DEST/${LANE}2.json'; \
     mv '$DEST/${LANE}2.xyz.partial' '$DEST/${LANE}2.xyz'; \
     if test -s '$DEST/A2.xyz' && test -s '$DEST/B2.xyz' && ! test -e '$DEST/.READY_WAVE2'; then \
       touch '$DEST/.READY_WAVE2'; \
       systemctl --user restart phonon-graphene-fd-thermal-refine-wave2.service; \
     fi"
touch "$SOURCE/RELAYED"
echo "[$(date -Is)] wave 2 lane $LANE relayed to 2060 through Tailscale"
