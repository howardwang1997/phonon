#!/usr/bin/env bash
# P0 graphene validation: dense DFPT line cuts through Gamma and K while the
# V100 GPU remains occupied by DFT-MD.  This script deliberately uses the
# CPU-only conda-QE build at low OS priority and writes into a new /data tree.
#
# Lane A: dg=0.01, k=64 (low-smearing converged reference)
# Lane B: dg=0.01/0.04/0.08, k=32 (k-grid check + melt line shapes)
#
# The q points lie on the Gamma--K ray.  In QE's ibrav=4 Cartesian convention
# (units 2*pi/a), K=(1/3,1/sqrt(3),0), hence q(t)=t*K.  t around zero resolves
# the Gamma cusp and t around one crosses the K cusp from both sides.
set -euo pipefail

LANE="${1:?usage: bash scripts/v100/run_graphene_dfpt_linecuts_p0.sh A|B}"
case "$LANE" in
  A) SETS=("0.01:64") ;;
  B) SETS=("0.01:32" "0.04:32" "0.08:32") ;;
  *) echo "unknown lane '$LANE' (expected A or B)" >&2; exit 2 ;;
esac

ROOT="${ROOT:-/root/phonon}"
QEBIN="${QEBIN:-/root/miniconda3/envs/qe/bin}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
PSEUDO="${PSEUDO:-$ROOT/pseudo}"
WORK="${WORK:-/data/p0_graphene_dfpt/$LANE}"
NP="${NP:-8}"
for exe in pw.x ph.x mpirun; do
  if [[ ! -x "$QEBIN/$exe" ]]; then
    echo "missing executable: $QEBIN/$exe" >&2
    exit 2
  fi
done
mkdir -p "$WORK"
cd "$WORK"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

LOG="$WORK/run.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== P0 graphene DFPT lane=$LANE start $(date -Is) ==="

# Exact lattice of results/sc_conv/graphene_sc8_dg*_phonopy.yaml.  Do not use
# the later relaxed-a=2.45758 data here: this line cut validates the 8x8 main set.
A_BOHR=4.648726286
C_OVER_A=6.097560976

# region:t.  Gamma needs only the outward ray; K is sampled on both sides.
QPOINTS=(
  "G:0.000" "G:0.015" "G:0.030" "G:0.050" "G:0.080"
  "K:0.920" "K:0.950" "K:0.970" "K:0.985" "K:1.000"
  "K:1.015" "K:1.030" "K:1.050" "K:1.080"
)

run_scf() {
  local setdir="$1"
  if grep -q "JOB DONE" "$setdir/scf.out" 2>/dev/null; then
    echo "[P0:$LANE] reuse $(basename "$setdir") SCF"
    return 0
  fi
  echo "[P0:$LANE] SCF $(basename "$setdir") $(date -Is)"
  (
    cd "$setdir"
    nice -n 10 "$QEBIN/mpirun" --allow-run-as-root -np "$NP" \
      "$QEBIN/pw.x" -in scf.in > scf.out 2>&1
  )
  grep -q "JOB DONE" "$setdir/scf.out"
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
    echo "[P0:$LANE] SCF FAILED dg=$DG k=$KGRID" >&2
    failures=$((failures + 1))
    continue
  fi

  for qt in "${QPOINTS[@]}"; do
    REGION="${qt%%:*}"
    T="${qt##*:}"
    SLUG="${REGION}_t${T//./p}"
    QDIR="$SETDIR/$SLUG"
    mkdir -p "$QDIR"
    read -r QX QY < <(awk -v t="$T" 'BEGIN {printf "%.10f %.10f\n", t/3.0, t/sqrt(3.0)}')
    cat > "$QDIR/ph.in" <<EOF
graphene P0 dense line cut, lane $LANE, dg=$DG, k=$KGRID, $REGION t=$T
&inputph
 prefix='gr', outdir='../tmp', fildyn='gr.dyn',
 tr2_ph=1.0d-14, ldisp=.false., trans=.true., epsil=.false., recover=.false.
/
$QX $QY 0.0
EOF
    if grep -q "JOB DONE" "$QDIR/ph.out" 2>/dev/null; then
      echo "[P0:$LANE] reuse dg=$DG k=$KGRID $REGION t=$T"
      continue
    fi
    echo "[P0:$LANE] PH dg=$DG k=$KGRID $REGION t=$T q=($QX,$QY) $(date -Is)"
    if ! (
      cd "$QDIR"
      nice -n 10 "$QEBIN/mpirun" --allow-run-as-root -np "$NP" \
        "$QEBIN/ph.x" -in ph.in > ph.out 2>&1
    ) || ! grep -q "JOB DONE" "$QDIR/ph.out"; then
      echo "[P0:$LANE] PH FAILED dg=$DG k=$KGRID $REGION t=$T" >&2
      failures=$((failures + 1))
    fi
  done
done

# Rebuild the aggregate CSV from completed outputs, so resume never duplicates rows.
"$CONDA" run --no-capture-output -n phonon python - "$LANE" "$WORK" <<'PY'
import csv
import re
import sys
from pathlib import Path

lane, work = sys.argv[1], Path(sys.argv[2])
rows = []
for setdir in sorted(work.glob("dg*_k*")):
    m = re.fullmatch(r"dg(.+)_k(\d+)", setdir.name)
    if not m:
        continue
    dg, kgrid = float(m.group(1)), int(m.group(2))
    for qdir in sorted(setdir.glob("[GK]_t*")):
        out = qdir / "ph.out"
        inp = qdir / "ph.in"
        if not out.is_file() or "JOB DONE" not in out.read_text(errors="ignore"):
            continue
        mi = re.search(r"\n\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$", inp.read_text())
        mt = re.fullmatch(r"([GK])_t(.+)", qdir.name)
        if not mi or not mt:
            continue
        region = mt.group(1)
        t = float(mt.group(2).replace("p", "."))
        # Match only a single numbered mode, e.g. ``freq ( 1)``.  At Gamma,
        # ph.x appends irrep summaries such as ``freq (1-2)``; the old broad
        # regex included those summaries and shifted the six-mode CSV columns.
        freqs = [
            float(x)
            for x in re.findall(
                r"^\s*freq\s*\(\s*\d+\s*\)\s*=.*?=\s*([-+0-9.Ee]+)\s*\[cm-1\]",
                out.read_text(errors="ignore"),
                flags=re.MULTILINE,
            )
        ]
        if len(freqs) != 6:
            continue
        rows.append([lane, dg, kgrid, region, t, float(mi.group(1)), float(mi.group(2)), *freqs])

rows.sort(key=lambda r: (r[1], r[2], r[3], r[4]))
expected = 14 if lane == "A" else 42
if len(rows) != expected:
    raise RuntimeError(
        f"lane {lane}: expected {expected} completed q points, found {len(rows)}"
    )
dest = work / f"dfpt_linecuts_{lane}.csv"
with dest.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["lane", "degauss_Ry", "kgrid", "region", "t_GK", "qx_2pia", "qy_2pia", *[f"f{i}_cm" for i in range(1, 7)]])
    writer.writerows(rows)
print(f"wrote {dest} with {len(rows)} completed q points")
PY

if (( failures == 0 )); then
  touch "$WORK/DONE"
  echo "=== P0 graphene DFPT lane=$LANE COMPLETE $(date -Is) ==="
else
  echo "=== P0 graphene DFPT lane=$LANE ended with $failures failures $(date -Is) ===" >&2
  exit 1
fi
