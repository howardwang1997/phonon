#!/usr/bin/env bash
# P1 matched-smearing audit: reproduce the experimental-lattice TMD fc2 set
# with cold smearing while holding lattice, supercell, k grid, cutoff and every
# other tmd_dft_fc2 setting fixed to the completed Fermi-Dirac campaign.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
OUT="$ROOT/results/tmd_exp_a_recovery"
SCRATCH="${SCRATCH:-/data/v100scratch}"
LOG="$OUT/cold_pairs_A.log"
DONE="$OUT/COLD_PAIRS_DONE"
FAILED="$OUT/COLD_PAIRS_FAILED"

cd "$ROOT"
export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
rm -f "$DONE"

verify_npz() {
  local path="$1" name="$2" lattice_a="$3" supercell="$4" tag="$5"
  "$CONDA" run -n phonon python -c '
import sys, numpy as np
path, name, lattice_a, supercell, tag = sys.argv[1:]
with np.load(path, allow_pickle=False) as data:
    assert str(data["material"].item()) == name
    assert str(data["tag"].item()) == tag
    assert np.isclose(float(data["a"]), float(lattice_a), rtol=0.0, atol=1e-10)
    assert int(data["supercell"]) == int(supercell)
    assert np.isclose(float(data["degauss"]), 0.005, rtol=0.0, atol=1e-12)
    assert str(data["smearing"].item()) == "cold"
    freq = np.asarray(data["frequencies"])
    assert freq.size and np.isfinite(freq).all()
' "$path" "$name" "$lattice_a" "$supercell" "$tag"
}

run_one() {
  local name="$1" lattice_a="$2" supercell="$3" tag="$4"
  local yaml="$OUT/${tag}_phonopy.yaml" npz="$OUT/disp_${tag}.npz"
  if [[ -s "$yaml" && -s "$npz" ]] && verify_npz "$npz" "$name" "$lattice_a" "$supercell" "$tag"; then
    echo "[$tag] validated artifacts exist -> skip"
    return 0
  fi
  echo "=== [cold-pair] $tag start $(date -Is) ==="
  if ! "$CONDA" run --no-capture-output -n phonon python \
      scripts/v100/tmd_dft_fc2.py \
      --name "$name" --a "$lattice_a" --tag "$tag" \
      --pw /root/gpupw.sh --nproc 1 --pseudo-dir /root/phonon/pseudo \
      --workdir results/tmd_exp_a_recovery --scratch "$SCRATCH" \
      --supercell "$supercell" --degauss 0.005 --smearing cold --force; then
    echo "=== [cold-pair] $tag calculation failed ===" >&2
    return 1
  fi
  if [[ ! -s "$yaml" || ! -s "$npz" ]]; then
    echo "=== [cold-pair] $tag missing output artifact ===" >&2
    return 1
  fi
  if ! verify_npz "$npz" "$name" "$lattice_a" "$supercell" "$tag"; then
    echo "=== [cold-pair] $tag provenance verification failed ===" >&2
    return 1
  fi
  echo "=== [cold-pair] $tag verified $(date -Is) ==="
}

# Run the most informative discrepancy first, followed by inexpensive 3x3
# systems, then the remaining expensive 4x4 TiSe2 point.
status=0
run_one 1T-VSe2 3.340 4 1T-VSe2_exp_a3.340_dg0.005_cold || status=1
run_one NbSe2 3.440 3 NbSe2_exp_a3.440_dg0.005_cold || status=1
run_one NbS2 3.320 3 NbS2_exp_a3.320_dg0.005_cold || status=1
run_one 2H-TaSe2 3.436 3 2H-TaSe2_exp_a3.436_dg0.005_cold || status=1
run_one 1T-TiSe2 3.540 4 1T-TiSe2_exp_a3.540_dg0.005_cold || status=1

if (( status != 0 )); then
  touch "$FAILED"
  echo "=== TMD cold-pair campaign FAILED $(date -Is) ===" >&2
  exit 1
fi
rm -f "$FAILED"
touch "$DONE"
echo "=== TMD cold-pair campaign COMPLETE $(date -Is) ==="
