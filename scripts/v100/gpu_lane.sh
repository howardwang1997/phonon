#!/usr/bin/env bash
# GPU lane for one box: keep the V100 GPU busy with FP64 pw.x work — for each of
# this box's assigned materials run  fc2 -> bands/chi(q) -> Path-P(CDW only).
# Every step is idempotent (skips if its output exists) so it's resumable and
# overlaps the CPU-lane EPW (GPU vs CPU). nproc 1 so it never starves the CPU lane.
#   bash scripts/v100/gpu_lane.sh A
set -uo pipefail
cd "$HOME/phonon"
BOX="${1:?need box A|B}"
CONDA="$HOME/miniconda3"
source "$CONDA/etc/profile.d/conda.sh"; conda activate phonon
export LD_LIBRARY_PATH="$CONDA/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
PY="$CONDA/envs/phonon/bin/python"
PW=$("$PY" -c "import sys;sys.path.insert(0,'scripts/v100');import tmd_common as t;print(t.load_config()['runtime']['pw'])")
PD=$("$PY" -c "import sys;sys.path.insert(0,'scripts/v100');import tmd_common as t;print(t.load_config()['runtime']['pseudo_dir'])")
mkdir -p results/v100/lanelogs
LOG="results/v100/lanelogs/gpu_${BOX}.log"
echo "=== GPU lane box $BOX start $(date) (pw=$PW pseudo=$PD) ===" | tee "$LOG"

run() { echo ">> $* $(date +%H:%M:%S)" | tee -a "$LOG"; "$@" >>"$LOG" 2>&1 \
        || echo "!! FAILED: $* (continuing)" | tee -a "$LOG"; }

"$PY" scripts/v100/tmd_common.py boxgpu "$BOX" | while read -r MAT CDW; do
  echo "===== material $MAT (cdw=$CDW) $(date +%H:%M:%S) =====" | tee -a "$LOG"
  run "$PY" scripts/v100/tmd_dft_fc2.py  --name "$MAT" --pw "$PW" --nproc 1 --pseudo-dir "$PD"
  run "$PY" scripts/v100/tmd_dft_bands.py --name "$MAT" --pw "$PW" --nproc 1 --pseudo-dir "$PD"
  if [ "$CDW" = "1" ]; then
    run "$PY" scripts/v100/tmd_path_p.py --name "$MAT" --pw "$PW" --pseudo-dir "$PD"
  fi
done
echo "=== GPU lane box $BOX DONE $(date) ===" | tee -a "$LOG"
touch "results/v100/GPU_LANE_${BOX}_DONE"
