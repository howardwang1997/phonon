#!/usr/bin/env bash
# Release-gated wrapper for one E1 lane.
set -euo pipefail

LANE="${1:?usage: run_graphene_e1_cross_degauss.sh A|B}"
case "$LANE" in A|B) ;; *) echo "invalid lane $LANE" >&2; exit 2 ;; esac
ROOT="${ROOT:-/root/phonon}"
INPUT_ROOT="${INPUT_ROOT:-/data/graphene_e1_cross_degauss/input}"
WORK="${WORK:-/data/graphene_e1_cross_degauss/$LANE}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
GPU_MAX_USED_MIB_BEFORE_START="${GPU_MAX_USED_MIB_BEFORE_START:-1024}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

test -s "$INPUT_ROOT/OPERATOR_GATE_PASS"
test -s "$INPUT_ROOT/e1_manifest.json"
test -s "$INPUT_ROOT/e1_configs.xyz"
available_kb="$(df -Pk /data | awk 'NR==2 {print $4}')"
if (( available_kb < 50 * 1024 * 1024 )); then
    echo "less than 50 GiB free on /data; refusing to start E1" >&2
    exit 1
fi

mkdir -p "$WORK"
exec 9>"$WORK/.lock"
if ! flock -n 9; then
    echo "E1 lane $LANE is already running"
    exit 0
fi
if [[ -s "$WORK/DONE" && -s "$WORK/summary.json" ]]; then
    echo "E1 lane $LANE already complete"
    exit 0
fi

# The V100 nodes are shared with unrelated training runs.  Do not start a QE
# GPU calculation while another compute process owns the card: that can make
# the first E1 task fail with an OOM and leave a misleading partial directory.
command -v nvidia-smi >/dev/null
gpu_wait_iteration=0
while true; do
    gpu_used_mib="$(nvidia-smi \
        --query-gpu=memory.used --format=csv,noheader,nounits | \
        awk 'NR == 1 {gsub(/[[:space:]]/, "", $1); print int($1)}')"
    gpu_compute_processes="$(nvidia-smi \
        --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | \
        awk 'NF {count++} END {print count + 0}')"
    if (( gpu_used_mib <= GPU_MAX_USED_MIB_BEFORE_START )) && \
       (( gpu_compute_processes == 0 )); then
        echo "[E1] lane=$LANE GPU available: used=${gpu_used_mib}MiB compute_processes=0"
        break
    fi
    if (( gpu_wait_iteration % 10 == 0 )); then
        echo "[E1] lane=$LANE waiting for GPU: used=${gpu_used_mib}MiB " \
             "compute_processes=${gpu_compute_processes} " \
             "threshold=${GPU_MAX_USED_MIB_BEFORE_START}MiB"
    fi
    gpu_wait_iteration=$((gpu_wait_iteration + 1))
    sleep "$GPU_POLL_SECONDS"
done

export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
"$CONDA" run --no-capture-output -n phonon python \
    "$ROOT/scripts/v100/graphene_e1_cross_degauss.py" \
    --manifest "$INPUT_ROOT/e1_manifest.json" \
    --configs "$INPUT_ROOT/e1_configs.xyz" \
    --lane "$LANE" --workdir "$WORK" \
    --pw /root/gpupw.sh --pseudo-dir "$ROOT/pseudo"
test -s "$WORK/summary.json"
date -Is > "$WORK/DONE.tmp"
mv "$WORK/DONE.tmp" "$WORK/DONE"
echo "[E1] lane=$LANE complete=$(date -Is)"
