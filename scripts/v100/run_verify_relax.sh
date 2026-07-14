#!/usr/bin/env bash
# Launch verify_relax_fc2.py for one box's assigned materials. Run in tmux ON the
# box (persists across ssh disconnect). Self-logs; check the LOG, not pgrep.
#   A) NbSe2 NbS2     B) 2H-TaSe2 1T-VSe2
set -uo pipefail
BOX="${1:?need A|B}"
cd "$HOME/phonon"
CONDA="$HOME/miniconda3"
source "$CONDA/etc/profile.d/conda.sh"; conda activate phonon
export LD_LIBRARY_PATH="$CONDA/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="$CONDA/envs/phonon/bin/python"
mkdir -p results/v100/verify_relax
LOG="results/v100/verify_relax/run_${BOX}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== verify_relax box $BOX start $(date) ==="
case "$BOX" in
  A) MATS="NbSe2 NbS2" ;;
  B) MATS="2H-TaSe2 1T-VSe2" ;;
  *) echo "bad box $BOX"; exit 1 ;;
esac
PW=/root/gpupw.sh; PD=/root/phonon/pseudo
for M in $MATS; do
  echo "===== $M $(date) ====="
  "$PY" scripts/v100/verify_relax_fc2.py --name "$M" --pw "$PW" --nproc 1 \
       --pseudo-dir "$PD" --smearings 0.005 0.020 || echo "!! $M FAILED"
done
echo "=== verify_relax box $BOX DONE $(date) ==="
touch "results/v100/verify_relax/DONE_${BOX}"
