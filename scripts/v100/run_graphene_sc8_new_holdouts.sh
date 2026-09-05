#!/usr/bin/env bash
# Experiment 4: three previously unseen 8x8 points for a clean post-P0 test.
# Settings intentionally match the original results/sc_conv 8x8 campaign:
# a=2.46 A, k=4, fd smearing, conv_thr=1e-9, plain/0.4, david, atomic+random.
set -uo pipefail

ROOT=/root/phonon
CONDA=/root/miniconda3/bin/conda
OUT="$ROOT/results/sc_conv"
SCRATCH=/data/v100scratch/graphene_sc8_new_holdouts
LOG=/data/graphene_sc8_new_holdouts/run.log
PAUSE=/data/phonon_offload/recovery_pause_B
mkdir -p "$OUT" "$SCRATCH" "$(dirname "$LOG")"
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

verify_point() {
  local dg="$1" tag="$2"
  local yaml="$OUT/${tag}_phonopy.yaml" npz="$OUT/disp_${tag}.npz"
  [[ -s "$yaml" && -s "$npz" ]] || return 1
  "$CONDA" run -n phonon python -c '
import sys, numpy as np, phonopy
npz, yaml, dg, tag = sys.argv[1:]
with np.load(npz, allow_pickle=False) as data:
    assert str(data["tag"].item()) == tag
    assert np.isclose(float(data["a"]), 2.46, rtol=0, atol=1e-12)
    assert np.isclose(float(data["degauss"]), float(dg), rtol=0, atol=1e-12)
    assert str(data["smearing"].item()) == "fd"
    assert int(data["supercell"]) == 8 and int(data["kpts"]) == 4
    assert np.isclose(float(data["conv_thr"]), 1e-9)
    assert str(data["mixing_mode"].item()) == "plain"
    assert str(data["diagonalization"].item()) == "david"
    freq = np.asarray(data["frequencies"])
    assert freq.size and np.isfinite(freq).all()
ph = phonopy.load(yaml, produce_fc=False)
assert tuple(np.diag(ph.supercell_matrix)) == (8, 8, 1)
assert ph.force_constants is not None and np.isfinite(ph.force_constants).all()
' "$npz" "$yaml" "$dg" "$tag"
}

run_point() {
  local dg="$1" tag="graphene_sc8_dg${1}"
  if verify_point "$dg" "$tag"; then
    echo "[$tag] validated artifacts exist -> skip"
    return 0
  fi
  local attempt=1
  while (( attempt <= 3 )); do
    echo "=== [experiment4] $tag attempt=$attempt start $(date -Is) ==="
    if "$CONDA" run --no-capture-output -n phonon python \
      scripts/m1_1b_graphene_dft.py \
      --pw /root/gpupw.sh --nproc 1 \
      --pseudo-dir /root/phonon/pseudo --pseudo C_ONCV_PBE-1.2.upf \
      --a 2.46 --supercell 8 --disp 0.03 \
      --ecutwfc 60 --ecutrho 240 --kpts 4 \
      --degauss "$dg" --smearing fd \
      --workdir results/sc_conv --tag "$tag" --scratch "$SCRATCH" \
      --conv-thr 1e-9 --mixing-beta 0.4 --mixing-mode plain \
      --electron-maxstep 200 --diagonalization david \
      --startingwfc atomic+random --disk-io low \
      --clean-scratch-on-success && verify_point "$dg" "$tag"; then
      echo "=== [experiment4] $tag VERIFIED $(date -Is) ==="
      return 0
    fi
    echo "=== [experiment4] $tag failed attempt=$attempt $(date -Is) ===" >&2
    attempt=$((attempt + 1))
  done
  return 1
}

resume_experiment3() {
  # The marker is ours and is removed whether experiment 4 succeeds or fails,
  # so the 600 K checkpointed lane cannot remain accidentally paused.
  rm -f "$PAUSE"
  systemctl start phonon-recovery-run@B.service || true
  echo "=== experiment 3 lane B resume requested $(date -Is) ==="
}
trap resume_experiment3 EXIT

status=0
run_point 0.015 || status=1
run_point 0.03 || status=1
run_point 0.06 || status=1
if (( status == 0 )); then
  touch /data/graphene_sc8_new_holdouts/DONE
  echo "=== experiment 4 three-point 8x8 campaign COMPLETE $(date -Is) ==="
else
  touch /data/graphene_sc8_new_holdouts/FAILED
  echo "=== experiment 4 campaign ended with failures $(date -Is) ===" >&2
fi
exit "$status"
