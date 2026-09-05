#!/usr/bin/env bash
# Relay a completed physical-FD thermal label set directly to the RTX 2060 over
# Tailscale.  The Mac-side monitor performs the same transfer as a backup.
set -euo pipefail

LANE="${1:?usage: run_graphene_fd_label_relay.sh A|B}"
case "$LANE" in A|B) ;; *) echo "unknown lane: $LANE" >&2; exit 2 ;; esac

SOURCE="/data/graphene_fd_thermal_labels/$LANE"
REMOTE="howardwang@100.105.21.7"
DEST="/home/howardwang/phonon/data/graphene_fd_thermal_labels"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
SCP=(scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)

while [[ ! -e "$SOURCE/DONE" ]]; do sleep 60; done
[[ -s "$SOURCE/summary.json" && -s "$SOURCE/summary.xyz" ]]
"${SSH[@]}" "$REMOTE" "mkdir -p '$DEST'"
"${SCP[@]}" "$SOURCE/summary.json" "$REMOTE:$DEST/$LANE.json.partial"
"${SCP[@]}" "$SOURCE/summary.xyz" "$REMOTE:$DEST/$LANE.xyz.partial"
"${SSH[@]}" "$REMOTE" \
    "mv '$DEST/$LANE.json.partial' '$DEST/$LANE.json'; \
     mv '$DEST/$LANE.xyz.partial' '$DEST/$LANE.xyz'; \
     if test -s '$DEST/A.xyz' && test -s '$DEST/B.xyz' && ! test -e '$DEST/.READY'; then \
       touch '$DEST/.READY'; \
       systemctl --user restart phonon-graphene-fd-thermal-finetune.service; \
     fi"
touch "$SOURCE/RELAYED"
echo "[$(date -Is)] lane $LANE relayed to 2060 through Tailscale"
