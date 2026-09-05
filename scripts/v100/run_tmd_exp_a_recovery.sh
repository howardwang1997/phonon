#!/usr/bin/env bash
# Recover the experimental-lattice TMD fc2 campaign.  Unlike the original
# ad-hoc wrappers, every CLI name is a real config key and DONE is written only
# after both the phonopy YAML and provenance-rich NPZ exist for every material.
set -euo pipefail

BOX="${1:?usage: bash scripts/v100/run_tmd_exp_a_recovery.sh A|B}"
cd /root/phonon
export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"

OUT="results/tmd_exp_a_recovery"
LOG="$OUT/recovery_${BOX}.log"
MARKER="$OUT/DONE_${BOX}"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

verify_npz() {
  local npz="$1" name="$2" lattice_a="$3" supercell="$4" tag="$5"
  /root/miniconda3/bin/conda run -n phonon python -c '
import sys, numpy as np
path, name, lattice_a, supercell, tag = sys.argv[1:]
with np.load(path, allow_pickle=False) as data:
    assert str(data["material"].item()) == name
    assert str(data["tag"].item()) == tag
    assert np.isclose(float(data["a"]), float(lattice_a), rtol=0.0, atol=1e-10)
    assert int(data["supercell"]) == int(supercell)
    assert np.isclose(float(data["degauss"]), 0.005, rtol=0.0, atol=1e-12)
    assert str(data["smearing"].item()) == "fd"
    freq = np.asarray(data["frequencies"])
    assert freq.size and np.isfinite(freq).all()
' "$npz" "$name" "$lattice_a" "$supercell" "$tag"
}

run_one() {
  local name="$1" lattice_a="$2" supercell="$3" tag="$4"
  local yaml="$OUT/${tag}_phonopy.yaml" npz="$OUT/disp_${tag}.npz"
  if [[ -s "$yaml" && -s "$npz" ]] && verify_npz "$npz" "$name" "$lattice_a" "$supercell" "$tag"; then
    echo "[$tag] validated artifacts exist -> skip"
    return 0
  fi
  echo "=== [$BOX] $tag start $(date -Is) ==="
  /root/miniconda3/bin/conda run --no-capture-output -n phonon python \
    scripts/v100/tmd_dft_fc2.py \
    --name "$name" --a "$lattice_a" --tag "$tag" \
    --pw /root/gpupw.sh --nproc 1 --pseudo-dir /root/phonon/pseudo \
    --workdir "$OUT" --scratch /data/v100scratch \
    --supercell "$supercell" --degauss 0.005 --smearing fd --force
  [[ -s "$yaml" && -s "$npz" ]]
  verify_npz "$npz" "$name" "$lattice_a" "$supercell" "$tag"
  echo "=== [$BOX] $tag verified $(date -Is) ==="
}

case "$BOX" in
  A)
    run_one NbSe2 3.440 3 NbSe2_exp_a3.440_dg0.005_fd
    run_one NbS2 3.320 3 NbS2_exp_a3.320_dg0.005_fd
    run_one 2H-TaSe2 3.436 3 2H-TaSe2_exp_a3.436_dg0.005_fd
    ;;
  B)
    run_one 1T-VSe2 3.340 4 1T-VSe2_exp_a3.340_dg0.005_fd
    run_one 1T-TiSe2 3.540 4 1T-TiSe2_exp_a3.540_dg0.005_fd
    ;;
  *)
    echo "unknown box '$BOX' (expected A or B)" >&2
    exit 2
    ;;
esac

touch "$MARKER"
echo "=== [$BOX] TMD experimental-a recovery COMPLETE $(date -Is) ==="
