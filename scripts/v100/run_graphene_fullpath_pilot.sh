#!/usr/bin/env bash
# P1 graphene pilot: direct-DFPT checks away from the already validated
# Gamma/K line-cut neighbourhoods.  Two electronic smearings share four
# points spanning Gamma-M, M, M-K and K-G.  The job is incremental and safe to
# restart: a q point is reused only when ph.x wrote JOB DONE and a non-empty
# dynamical matrix.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
QEBIN="${QEBIN:-/root/miniconda3/envs/qe/bin}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
PSEUDO="${PSEUDO:-$ROOT/pseudo}"
WORK="${WORK:-/data/graphene_fullpath_pilot}"
OUT="$ROOT/results/p1_graphene_fullpath_pilot"
NP="${NP:-8}"
LOG="$WORK/run.log"
DONE="$WORK/DONE"
FAILED="$WORK/FAILED"

for exe in pw.x ph.x mpirun; do
  if [[ ! -x "$QEBIN/$exe" ]]; then
    echo "missing executable: $QEBIN/$exe" >&2
    exit 2
  fi
done

mkdir -p "$WORK" "$OUT"
cd "$WORK"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec > >(tee -a "$LOG") 2>&1
rm -f "$DONE"

echo "=== graphene full-path pilot start $(date -Is) ==="

# Exact primitive cell and electronic settings used by the frozen 8x8 method
# and final direct-DFPT holdout.  k=32 is already converged against k=64 to
# 0.0213 cm^-1 on the final low-smearing K subset.
A_BOHR=4.648726286
C_OVER_A=6.097560976
SETS=("0.013:32" "0.055:32")

# label:h:k in reciprocal reduced coordinates.  Conversion to QE's Cartesian
# 2*pi/a convention for ibrav=4 is (qx,qy)=(h,(h+2k)/sqrt(3)).
QPOINTS=(
  "GM_mid:0.2500000000:0.0000000000"
  "M:0.5000000000:0.0000000000"
  "MK_mid:0.4166666667:0.1666666667"
  "KG_mid:0.1666666667:0.1666666667"
)

run_scf() {
  local setdir="$1"
  if grep -q "JOB DONE" "$setdir/scf.out" 2>/dev/null; then
    echo "[pilot] reuse $(basename "$setdir") SCF"
    return 0
  fi
  echo "[pilot] SCF $(basename "$setdir") $(date -Is)"
  (
    cd "$setdir"
    nice -n 10 "$QEBIN/mpirun" --allow-run-as-root -np "$NP" \
      "$QEBIN/pw.x" -in scf.in > scf.out 2>&1
  )
  grep -q "JOB DONE" "$setdir/scf.out"
}

run_ph() {
  local qdir="$1" dg="$2" kgrid="$3" label="$4"
  if grep -q "JOB DONE" "$qdir/ph.out" 2>/dev/null && [[ -s "$qdir/gr.dyn" ]]; then
    echo "[pilot] reuse dg=$dg k=$kgrid $label"
    return 0
  fi
  local attempt
  for attempt in 1 2 3; do
    echo "[pilot] PH dg=$dg k=$kgrid $label attempt=$attempt $(date -Is)"
    if (
      cd "$qdir"
      nice -n 10 "$QEBIN/mpirun" --allow-run-as-root -np "$NP" \
        "$QEBIN/ph.x" -in ph.in > ph.out 2>&1
    ) && grep -q "JOB DONE" "$qdir/ph.out" && [[ -s "$qdir/gr.dyn" ]]; then
      return 0
    fi
  done
  echo "[pilot] PH FAILED dg=$dg k=$kgrid $label" >&2
  return 1
}

failures=0
for spec in "${SETS[@]}"; do
  DG="${spec%%:*}"
  KGRID="${spec##*:}"
  SETDIR="$WORK/dg${DG}_k${KGRID}"
  mkdir -p "$SETDIR/tmp"
  cat > "$SETDIR/scf.in" <<EOF
&control
 calculation='scf', prefix='gr', outdir='./tmp', pseudo_dir='$PSEUDO',
 verbosity='low', disk_io='low'
/
&system
 ibrav=4, celldm(1)=$A_BOHR, celldm(3)=$C_OVER_A, nat=2, ntyp=1,
 ecutwfc=60, ecutrho=240, occupations='smearing', smearing='fd', degauss=$DG
/
&electrons
 conv_thr=1.0d-11, mixing_beta=0.5, electron_maxstep=200
/
ATOMIC_SPECIES
 C 12.011 C_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
 C 0.000000000 0.000000000 0.0
 C 0.333333333 0.666666667 0.0
K_POINTS automatic
 $KGRID $KGRID 1 0 0 0
EOF

  if ! run_scf "$SETDIR"; then
    echo "[pilot] SCF FAILED dg=$DG k=$KGRID" >&2
    failures=$((failures + 1))
    continue
  fi

  for point in "${QPOINTS[@]}"; do
    IFS=: read -r LABEL H K <<< "$point"
    QDIR="$SETDIR/$LABEL"
    mkdir -p "$QDIR"
    read -r QX QY < <(awk -v h="$H" -v k="$K" \
      'BEGIN {printf "%.10f %.10f\n", h, (h+2*k)/sqrt(3.0)}')
    cat > "$QDIR/ph.in" <<EOF
graphene P1 full-path pilot, dg=$DG, k=$KGRID, $LABEL, reduced=($H,$K,0)
&inputph
 prefix='gr', outdir='../tmp', fildyn='gr.dyn',
 tr2_ph=1.0d-14, ldisp=.false., trans=.true., epsil=.false., recover=.false.
/
$QX $QY 0.0
EOF
    run_ph "$QDIR" "$DG" "$KGRID" "$LABEL" || failures=$((failures + 1))
  done
done

if (( failures > 0 )); then
  touch "$FAILED"
  echo "=== graphene full-path pilot FAILED ($failures points) $(date -Is) ===" >&2
  exit 1
fi

# Rebuild a strict aggregate from completed outputs on every resume.
"$CONDA" run --no-capture-output -n phonon python - "$WORK" "$OUT" <<'PY'
import csv
import re
import sys
from pathlib import Path

import numpy as np

work, out = map(Path, sys.argv[1:])
expected_labels = {"GM_mid", "M", "MK_mid", "KG_mid"}
rows = []
for setdir in sorted(work.glob("dg*_k*")):
    match = re.fullmatch(r"dg(.+)_k(\d+)", setdir.name)
    if not match:
        continue
    dg, kgrid = float(match.group(1)), int(match.group(2))
    for qdir in sorted(path for path in setdir.iterdir() if path.is_dir() and path.name != "tmp"):
        ph_in, ph_out, dyn = qdir / "ph.in", qdir / "ph.out", qdir / "gr.dyn"
        if not (ph_in.is_file() and ph_out.is_file() and dyn.is_file()):
            continue
        if "JOB DONE" not in ph_out.read_text(errors="ignore") or dyn.stat().st_size == 0:
            continue
        title = ph_in.read_text(errors="ignore").splitlines()[0]
        reduced = re.search(r"reduced=\(([-+0-9.]+),([-+0-9.]+),0\)", title)
        qline = ph_in.read_text(errors="ignore").strip().splitlines()[-1].split()
        frequencies = [
            float(value)
            for value in re.findall(
                r"^\s*freq\s*\(\s*\d+\s*\)\s*=.*?=\s*([-+0-9.Ee]+)\s*\[cm-1\]",
                ph_out.read_text(errors="ignore"),
                flags=re.MULTILINE,
            )
        ]
        if reduced is None or len(qline) != 3 or len(frequencies) != 6:
            raise RuntimeError(f"could not parse completed point {qdir}")
        rows.append([
            dg, kgrid, qdir.name,
            float(reduced.group(1)), float(reduced.group(2)),
            float(qline[0]), float(qline[1]), *frequencies,
        ])

keys = {(row[0], row[2]) for row in rows}
expected = {(dg, label) for dg in (0.013, 0.055) for label in expected_labels}
if keys != expected or len(rows) != 8:
    raise RuntimeError(f"pilot incomplete; missing={sorted(expected-keys)}, extra={sorted(keys-expected)}")
if not np.isfinite(np.asarray([row[7:] for row in rows], float)).all():
    raise RuntimeError("non-finite pilot frequencies")
rows.sort(key=lambda row: (row[0], row[2]))
dest = out / "graphene_fullpath_pilot_dfpt.csv"
with dest.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "degauss_Ry", "kgrid", "label", "q_reduced_h", "q_reduced_k",
        "qx_2pia", "qy_2pia", *[f"f{i}_cm" for i in range(1, 7)],
    ])
    writer.writerows(rows)
print(f"wrote {dest} with {len(rows)} points")
PY

rm -f "$FAILED"
touch "$DONE"
echo "=== graphene full-path pilot COMPLETE $(date -Is) ==="
