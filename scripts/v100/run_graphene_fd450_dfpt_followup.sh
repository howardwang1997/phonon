#!/usr/bin/env bash
# Gate FD450_CONV, run FD450_LINE only if k=120->144 is converged, and relay results.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
BASE="${WORK_BASE:-/data/graphene_physical_fd_dfpt}"
CONV="$BASE/campaigns/FD450_CONV"
LINE="$BASE/campaigns/FD450_LINE"
CHECK="$CONV/convergence_acceptance.json"
REMOTE="howardwang@100.105.21.7"
DEST="/home/howardwang/phonon/results/graphene_physical_fd_dfpt/campaigns"
SSH=(env -u LD_LIBRARY_PATH ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
SCP=(env -u LD_LIBRARY_PATH scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)

relay_campaign() {
    local campaign="$1"
    local source="$BASE/campaigns/$campaign"
    local filename="graphene_${campaign}_dfpt.csv"
    "${SSH[@]}" "$REMOTE" "mkdir -p '$DEST/$campaign'"
    for path in "$source/$filename" "$source/run.log"; do
        local name
        name="$(basename "$path")"
        "${SCP[@]}" "$path" "$REMOTE:$DEST/$campaign/$name.partial"
        "${SSH[@]}" "$REMOTE" \
            "mv '$DEST/$campaign/$name.partial' '$DEST/$campaign/$name'"
    done
}

while [[ ! -e "$CONV/DONE" ]]; do sleep 60; done
CSV="$CONV/graphene_FD450_CONV_dfpt.csv"
test -s "$CSV"
if "$CONDA" run --no-capture-output -n phonon python \
    "$ROOT/scripts/smearing_kink/check_graphene_fd450_dfpt_convergence.py" \
    --csv "$CSV" --degauss 0.00285013035 --lower-k 120 --upper-k 144 \
    --threshold-cm-1 1 --output "$CHECK"; then
    touch "$CONV/PASSED_KGRID_GATE"
else
    touch "$CONV/BLOCKED_KGRID_GATE"
fi

relay_campaign FD450_CONV
"${SCP[@]}" "$CHECK" "$REMOTE:$DEST/FD450_CONV/convergence_acceptance.json.partial"
"${SSH[@]}" "$REMOTE" \
    "mv '$DEST/FD450_CONV/convergence_acceptance.json.partial' '$DEST/FD450_CONV/convergence_acceptance.json'; \
     touch '$DEST/FD450_CONV/REMOTE_DONE'"
touch "$CONV/RELAYED_TO_2060"

if [[ -e "$CONV/BLOCKED_KGRID_GATE" ]]; then
    echo "FD450 k-grid convergence failed; dense line remains locked"
    exit 0
fi

systemctl reset-failed phonon-graphene-physical-fd@FD450_LINE.service 2>/dev/null || true
systemctl start phonon-graphene-physical-fd@FD450_LINE.service
while [[ ! -e "$LINE/DONE" ]]; do
    state="$(systemctl is-active phonon-graphene-physical-fd@FD450_LINE.service 2>/dev/null || true)"
    if [[ "$state" != active && "$state" != activating ]]; then
        systemctl restart phonon-graphene-physical-fd@FD450_LINE.service
    fi
    sleep 60
done
relay_campaign FD450_LINE
"${SSH[@]}" "$REMOTE" "touch '$DEST/FD450_LINE/REMOTE_DONE'"
touch "$LINE/RELAYED_TO_2060"
echo "FD450 convergence and dense-line follow-up complete"
