#!/usr/bin/env bash
# Compute one fixed 450 K on-policy DFT force-label shard after the sampling gate.
set -euo pipefail

SHARD="${1:?usage: run_graphene_fd_transferability_p4b_shard.sh A|B}"
ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
REMOTE="howardwang@100.105.21.7"
REMOTE_ROOT="/home/howardwang/phonon"
REMOTE_ON_POLICY="$REMOTE_ROOT/results/graphene_fd_transferability/T450/on_policy"
WORK="${WORK:-/data/graphene_fd_transferability_p4b/shard_${SHARD}}"
INPUTS="$WORK/inputs"
DEST="$REMOTE_ROOT/data/graphene_fd_transferability/T450/on_policy/dft_force_labels/shard_${SHARD}"
LOG="$WORK/run.log"
ALL_INDICES="3,9,15,21,27,33,39,45,51,57,63,69,75,81,87,93,99,105,111,117"
FIRST_INDICES="3,9,15,21,27,33,39,45,51,57"
LAST_INDICES="63,69,75,81,87,93,99,105,111,117"
SSH=(env -u LD_LIBRARY_PATH ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
SCP=(env -u LD_LIBRARY_PATH scp -p -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)

case "$SHARD" in
    A) LANE_SPECS=("0:$ALL_INDICES" "2:$FIRST_INDICES") ;;
    B) LANE_SPECS=("1:$ALL_INDICES" "2:$LAST_INDICES") ;;
    *) echo "SHARD must be A or B" >&2; exit 2 ;;
esac

mkdir -p "$INPUTS"
exec 9>"$WORK/.shard_${SHARD}.lock"
if ! flock -n 9; then
    echo "450 K force-label shard $SHARD is already running; refusing a duplicate"
    exit 0
fi
if [[ -e "$WORK/SHARD_${SHARD}_DONE" && -e "$WORK/RELAYED_SHARD_${SHARD}_TO_2060" ]]; then
    exit 0
fi
exec > >(tee -a "$LOG") 2>&1

wait_round=0
until "${SSH[@]}" "$REMOTE" "test -e '$REMOTE_ON_POLICY/READY_DFT_FORCE_LABELS'"; do
    if "${SSH[@]}" "$REMOTE" \
        "test -e '$REMOTE_ON_POLICY/BLOCKED_SAMPLING_POINT_N3000' -o -e '$REMOTE_ON_POLICY/BLOCKED_SAMPLING_BOOTSTRAP_N3000'"; then
        echo "450 K n=3000 sampling gate failed; shard $SHARD remains locked"
        exit 0
    fi
    if (( wait_round % 10 == 0 )); then
        echo "waiting for the final 450 K n=3000 bootstrap release: $(date -Is)"
    fi
    wait_round=$((wait_round + 1))
    sleep 60
done

wait_round=0
while [[ ! -e "$WORK/SHARD_${SHARD}_DONE" ]] && \
      (pgrep -x pw.x >/dev/null || pgrep -x ph.x >/dev/null); do
    if (( wait_round % 10 == 0 )); then
        echo "waiting for the active QE calculation on this node: $(date -Is)"
    fi
    wait_round=$((wait_round + 1))
    sleep 60
done

fetch() {
    local source="$1"
    local destination="$2"
    "${SCP[@]}" "$REMOTE:$source" "$destination.partial"
    mv "$destination.partial" "$destination"
}

if [[ ! -e "$WORK/SHARD_${SHARD}_DONE" ]]; then
    echo "=== graphene transferability P4b shard=$SHARD start $(date -Is) ==="
    fetch "$REMOTE_ON_POLICY/READY_DFT_FORCE_LABELS" "$INPUTS/READY_DFT_FORCE_LABELS"
    fetch "$REMOTE_ON_POLICY/sampling_acceptance_n3000.json" "$INPUTS/sampling_acceptance.json"
    fetch "$REMOTE_ON_POLICY/sampling_bootstrap_acceptance_n3000.json" "$INPUTS/sampling_bootstrap.json"
    fetch "$REMOTE_ROOT/results/graphene_fd_transferability/freeze_manifest.json" "$INPUTS/freeze_manifest.json"
    SNAPSHOT_ARGS=()
    for specification in "${LANE_SPECS[@]}"; do
        seed="${specification%%:*}"
        destination="$INPUTS/seed${seed}_snapshots.npz"
        case "$seed" in
            0) source="$REMOTE_ON_POLICY/checkpoints/seed0_dt0p5/T450/snapshots.npz" ;;
            1) source="$REMOTE_ON_POLICY/checkpoints/seed1_dt0p25/T450/snapshots.npz" ;;
            2) source="$REMOTE_ON_POLICY/checkpoints/seed2_dt0p5/T450/snapshots.npz" ;;
        esac
        fetch "$source" "$destination"
        SNAPSHOT_ARGS+=(--snapshot "$seed=$destination")
    done

    export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
    "$CONDA" run --no-capture-output -n phonon python \
        "$ROOT/scripts/smearing_kink/validate_graphene_fd_p4b_inputs.py" \
        --freeze-manifest "$INPUTS/freeze_manifest.json" \
        --sampling-acceptance "$INPUTS/sampling_acceptance.json" \
        --sampling-bootstrap "$INPUTS/sampling_bootstrap.json" \
        "${SNAPSHOT_ARGS[@]}" --shard "$SHARD" --output "$WORK/input_manifest.json"

    for specification in "${LANE_SPECS[@]}"; do
        seed="${specification%%:*}"
        indices="${specification#*:}"
        lane="$WORK/seed${seed}"
        mkdir -p "$lane/labels"
        "$CONDA" run --no-capture-output -n phonon python \
            "$ROOT/scripts/v100/graphene_fd_force_convergence.py" \
            --snapshots "$INPUTS/seed${seed}_snapshots.npz" \
            --indices "$indices" --validation-indices "$indices" \
            --trajectory-seed "$seed" --kgrids 8 --disk-io none \
            --degauss 0.00285013035 --lattice-temperature 450 \
            --pw /root/gpupw.sh --pseudo-dir "$ROOT/pseudo" \
            --workdir "$lane" --output "$lane/summary.json"
        test -s "$lane/summary.json"
        test -s "$lane/summary.xyz"
    done
    touch "$WORK/SHARD_${SHARD}_DONE"
else
    echo "shard $SHARD labels already complete; retrying Tailscale relay only"
fi

"${SSH[@]}" "$REMOTE" "mkdir -p '$DEST'"
for source in "$WORK/input_manifest.json" "$LOG"; do
    filename="$(basename "$source")"
    "${SCP[@]}" "$source" "$REMOTE:$DEST/$filename.partial"
    "${SSH[@]}" "$REMOTE" "mv '$DEST/$filename.partial' '$DEST/$filename'"
done
for specification in "${LANE_SPECS[@]}"; do
    seed="${specification%%:*}"
    "${SSH[@]}" "$REMOTE" "mkdir -p '$DEST/seed${seed}'"
    for source in "$WORK/seed${seed}/summary.json" "$WORK/seed${seed}/summary.xyz"; do
        filename="$(basename "$source")"
        "${SCP[@]}" "$source" "$REMOTE:$DEST/seed${seed}/$filename.partial"
        "${SSH[@]}" "$REMOTE" "mv '$DEST/seed${seed}/$filename.partial' '$DEST/seed${seed}/$filename'"
    done
done
"${SSH[@]}" "$REMOTE" "touch '$DEST/RAW_READY'"
touch "$WORK/RELAYED_SHARD_${SHARD}_TO_2060"
echo "=== graphene transferability P4b shard=$SHARD COMPLETE $(date -Is) ==="
