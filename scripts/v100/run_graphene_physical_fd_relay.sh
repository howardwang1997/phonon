#!/usr/bin/env bash
# Relay completed dense-line and zero-smearing DFPT summaries directly to the
# RTX 2060 through its Tailscale address.
set -euo pipefail

LANE="${1:?usage: run_graphene_physical_fd_relay.sh A|B}"
case "$LANE" in
  A) CAMPAIGNS=(FD300_LINE) ;;
  B) CAMPAIGNS=(FD600_LINE FD0_SCAN) ;;
  *) echo "unknown lane: $LANE" >&2; exit 2 ;;
esac
REMOTE="howardwang@100.105.21.7"
SOURCE=/data/graphene_physical_fd_dfpt/campaigns
DEST=/home/howardwang/phonon/results/graphene_physical_fd_dfpt/campaigns
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
SCP=(scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)

for campaign in "${CAMPAIGNS[@]}"; do
    while [[ ! -e "$SOURCE/$campaign/DONE" ]]; do sleep 60; done
    filename="graphene_${campaign}_dfpt.csv"
    [[ -s "$SOURCE/$campaign/$filename" ]]
    "${SSH[@]}" "$REMOTE" "mkdir -p '$DEST/$campaign'"
    "${SCP[@]}" "$SOURCE/$campaign/$filename" \
        "$REMOTE:$DEST/$campaign/$filename.partial"
    "${SCP[@]}" "$SOURCE/$campaign/run.log" \
        "$REMOTE:$DEST/$campaign/run.log.partial"
    "${SSH[@]}" "$REMOTE" \
        "mv '$DEST/$campaign/$filename.partial' '$DEST/$campaign/$filename'; \
         mv '$DEST/$campaign/run.log.partial' '$DEST/$campaign/run.log'; \
         touch '$DEST/$campaign/REMOTE_DONE'; \
         systemctl --user reset-failed phonon-graphene-physical-fd-postprocess.service 2>/dev/null || true; \
         systemctl --user start phonon-graphene-physical-fd-postprocess.service"
    touch "$SOURCE/$campaign/RELAYED_TO_2060"
    echo "[$(date -Is)] $campaign relayed to 2060 through Tailscale"
done
