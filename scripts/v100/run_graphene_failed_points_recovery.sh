#!/usr/bin/env bash
# Recover graphene points that were left missing by earlier V100 campaigns.
set -euo pipefail

BOX="${1:?usage: bash scripts/v100/run_graphene_failed_points_recovery.sh A|B}"
cd /root/phonon
export LD_LIBRARY_PATH="/root/miniconda3/envs/phonon/lib:${LD_LIBRARY_PATH:-}"

verify_npz() {
  local npz="$1" lattice_a="$2" degauss="$3" tag="$4"
  /root/miniconda3/bin/conda run -n phonon python -c '
import sys, numpy as np
path, lattice_a, degauss, tag = sys.argv[1:]
with np.load(path, allow_pickle=False) as data:
    assert str(data["tag"].item()) == tag
    assert np.isclose(float(data["a"]), float(lattice_a), rtol=0.0, atol=1e-10)
    assert np.isclose(float(data["degauss"]), float(degauss), rtol=0.0, atol=1e-12)
    assert str(data["smearing"].item()) == "fd"
    assert str(data["diagonalization"].item()) == "cg"
    assert str(data["mixing_mode"].item()) == "local-TF"
    assert int(data["electron_maxstep"]) == 500
    freq = np.asarray(data["frequencies"])
    assert freq.size and np.isfinite(freq).all()
' "$npz" "$lattice_a" "$degauss" "$tag"
}

run_point() {
  local lattice_a="$1" degauss="$2" outdir="$3" tag="$4"
  local yaml="$outdir/${tag}_phonopy.yaml" npz="$outdir/disp_${tag}.npz"
  if [[ -s "$yaml" && -s "$npz" ]] && verify_npz "$npz" "$lattice_a" "$degauss" "$tag"; then
    echo "[$tag] validated artifacts exist -> skip"
    return 0
  fi
  echo "=== [$BOX] $tag start $(date -Is) ==="
  /root/miniconda3/bin/conda run --no-capture-output -n phonon python \
    scripts/m1_1b_graphene_dft.py \
    --pw /root/gpupw.sh --nproc 1 --pseudo-dir /root/phonon/pseudo \
    --pseudo C_ONCV_PBE-1.2.upf --a "$lattice_a" --supercell 6 \
    --ecutwfc 60 --ecutrho 240 --kpts 6 --degauss "$degauss" --smearing fd \
    --workdir "$outdir" --tag "$tag" --scratch /data/v100scratch/graphene_recovery \
    --conv-thr 1e-8 --mixing-beta 0.20 --mixing-mode local-TF \
    --electron-maxstep 500 --diagonalization cg --startingwfc atomic \
    --diago-thr-init 1e-4 --disk-io none --clean-scratch-on-success
  [[ -s "$yaml" && -s "$npz" ]]
  verify_npz "$npz" "$lattice_a" "$degauss" "$tag"
  echo "=== [$BOX] $tag verified $(date -Is) ==="
}

case "$BOX" in
  A)
    OUT="results/vq_surface"
    mkdir -p "$OUT"
    exec > >(tee -a "$OUT/recovery_a2.50.log") 2>&1
    run_point 2.50 0.002 "$OUT" gr_a2.50_dg0.002
    run_point 2.50 0.005 "$OUT" gr_a2.50_dg0.005
    run_point 2.50 0.010 "$OUT" gr_a2.50_dg0.010
    run_point 2.50 0.020 "$OUT" gr_a2.50_dg0.020
    run_point 2.50 0.040 "$OUT" gr_a2.50_dg0.040
    touch "$OUT/RECOVERY_A_DONE"
    ;;
  B)
    OUT="results/graphene_kohn_fd"
    mkdir -p "$OUT"
    exec > >(tee -a "$OUT/recovery_dg0.20.log") 2>&1
    run_point 2.457580 0.20 "$OUT" graphene_sc6_dg0.20
    touch "$OUT/RECOVERY_DG020_DONE"
    ;;
  *)
    echo "unknown box '$BOX' (expected A or B)" >&2
    exit 2
    ;;
esac

echo "=== [$BOX] graphene failed-point recovery COMPLETE $(date -Is) ==="
