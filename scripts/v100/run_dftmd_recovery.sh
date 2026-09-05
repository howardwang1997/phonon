#!/usr/bin/env bash
# Resume the two legacy DFT-MD lanes from their last QE input geometry, then
# checkpoint every MD step so SSH/SIGHUP can no longer erase days of work.
set -euo pipefail

BOX="${1:?usage: bash scripts/v100/run_dftmd_recovery.sh A|B}"
cd /root/phonon
export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"

case "$BOX" in
  A)
    TEMP=300
    LEGACY="/root/gr_dftmd/T300/espresso.pwi"
    TAG="graphene_dft_tdep_300K_recovery"
    ;;
  B)
    TEMP=600
    LEGACY="/root/gr_dftmd/T600/espresso.pwi"
    TAG="graphene_dft_tdep_600K_recovery"
    ;;
  *)
    echo "unknown box '$BOX' (expected A or B)" >&2
    exit 2
    ;;
esac

[[ -s "$LEGACY" ]]
WORKROOT="/data/gr_dftmd_recovery/$BOX"
OUT_NPZ="results/td_phonon/${TAG}.npz"
OUT_CSV="results/td_phonon/${TAG}.csv"
LOG="$WORKROOT/recovery.log"
mkdir -p "$WORKROOT" results/td_phonon
exec > >(tee -a "$LOG") 2>&1

echo "=== [$BOX] checkpointed DFT-MD recovery start $(date -Is) ==="
/root/miniconda3/bin/conda run --no-capture-output -n phonon python \
  scripts/smearing_kink/dft_md_tdep.py \
  --phonopy results/vq_kink6/graphene_sc6_dg0.005_phonopy.yaml \
  --pw /root/gpupw.sh --pseudo-dir /root/phonon/pseudo \
  --workroot "$WORKROOT" --supercell 6,6,1 \
  --nsnap 60 --equil 100 --stride 20 --dt 1.0 --cutoff2 6.0 \
  --kpts 4,4,1 --degauss 0.005 --temperatures "$TEMP" --tag "$TAG" \
  --initial-structure "$LEGACY" --checkpoint-every 1 --resume \
  --provenance-note "legacy interrupted 500-step equilibration; seeded from last QE input and rethermalized for 100 steps"

[[ -s "$OUT_NPZ" && -s "$OUT_CSV" ]]
/root/miniconda3/bin/conda run -n phonon python -c '
import sys, numpy as np
result, snapshots, state, temp = sys.argv[1:]
with np.load(result, allow_pickle=False) as data:
    freq = np.asarray(data[f"T{temp}_freq"])
    assert freq.size and np.isfinite(freq).all()
with np.load(snapshots, allow_pickle=False) as data:
    assert np.asarray(data["positions"]).shape[0] == 60
    assert np.asarray(data["forces"]).shape[0] == 60
with np.load(state, allow_pickle=False) as data:
    assert int(data["equil_done"]) == 100
    assert int(data["sample_done"]) == 60
    assert int(data["stride_done"]) == 0
' "$OUT_NPZ" "$WORKROOT/T${TEMP}/snaps_T${TEMP}.npz" \
  "$WORKROOT/T${TEMP}/md_checkpoint/state.npz" "$TEMP"
touch "$WORKROOT/DONE"
echo "=== [$BOX] checkpointed DFT-MD recovery COMPLETE $(date -Is) ==="
