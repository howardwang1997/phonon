#!/usr/bin/env bash
# Restartable direct-DFPT campaigns for the graphene q-space Kohn correction.
#
# Development campaigns may run immediately:
#   DEV_A -> dg=0.02, k=32, original development q grid
#   DEV_B -> dg=0.06, k=32, original development q grid
#
# Final holdout campaigns are guarded by freeze_manifest.json:
#   HOLD_A -> dg=0.013/0.055, k=32, shifted holdout q grid
#   HOLD_B -> dg=0.027, k=32 + dg=0.013, k=64 five-point K check
set -euo pipefail

CAMPAIGN="${1:?usage: bash scripts/v100/run_graphene_qspace_dfpt.sh DEV_A|DEV_B|HOLD_A|HOLD_B}"
case "$CAMPAIGN" in
  DEV_A)
    SETS=("0.02:32")
    EXPECTED_ROWS=14
    HOLDOUT=0
    ;;
  DEV_B)
    SETS=("0.06:32")
    EXPECTED_ROWS=14
    HOLDOUT=0
    ;;
  HOLD_A)
    SETS=("0.013:32" "0.055:32")
    EXPECTED_ROWS=28
    HOLDOUT=1
    ;;
  HOLD_B)
    SETS=("0.027:32" "0.013:64")
    EXPECTED_ROWS=19
    HOLDOUT=1
    ;;
  *)
    echo "unknown campaign '$CAMPAIGN'" >&2
    exit 2
    ;;
esac

ROOT="${ROOT:-/root/phonon}"
QEBIN="${QEBIN:-/root/miniconda3/envs/qe/bin}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
PSEUDO="${PSEUDO:-$ROOT/pseudo}"
WORK="${WORK:-/data/graphene_qspace_dfpt/$CAMPAIGN}"
NP="${NP:-8}"
for exe in pw.x ph.x mpirun; do
  if [[ ! -x "$QEBIN/$exe" ]]; then
    echo "missing executable: $QEBIN/$exe" >&2
    exit 2
  fi
done

if (( HOLDOUT )); then
  FREEZE="$ROOT/results/p0_graphene_qspace/freeze_manifest.json"
  [[ -s "$FREEZE" ]] || {
    echo "refusing final holdout: missing $FREEZE" >&2
    exit 3
  }
  "$CONDA" run -n phonon python -c '
import json, sys
path, campaign = sys.argv[1:]
data = json.load(open(path))
assert data["status"] == "frozen_before_holdout"
assert campaign in data["authorized_holdout_campaigns"]
assert data["holdout_evaluated"] is False
' "$FREEZE" "$CAMPAIGN"
fi

mkdir -p "$WORK"
cd "$WORK"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

LOG="$WORK/run.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== graphene q-space DFPT campaign=$CAMPAIGN start $(date -Is) ==="

# Exact lattice used by the 8x8 finite-displacement and P0 direct-DFPT data.
A_BOHR=4.648726286
C_OVER_A=6.097560976

QPOINTS_DEVELOPMENT=(
  "G:0.000" "G:0.015" "G:0.030" "G:0.050" "G:0.080"
  "K:0.920" "K:0.950" "K:0.970" "K:0.985" "K:1.000"
  "K:1.015" "K:1.030" "K:1.050" "K:1.080"
)
QPOINTS_HOLDOUT=(
  "G:0.000" "G:0.012" "G:0.025" "G:0.045" "G:0.075"
  "K:0.925" "K:0.955" "K:0.975" "K:0.990" "K:1.000"
  "K:1.010" "K:1.025" "K:1.045" "K:1.075"
)
QPOINTS_K64=(
  "K:0.975" "K:0.990" "K:1.000" "K:1.010" "K:1.025"
)

run_scf() {
  local setdir="$1"
  if grep -q "JOB DONE" "$setdir/scf.out" 2>/dev/null; then
    echo "[$CAMPAIGN] reuse $(basename "$setdir") SCF"
    return 0
  fi
  echo "[$CAMPAIGN] SCF $(basename "$setdir") $(date -Is)"
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
    echo "[$CAMPAIGN] SCF FAILED dg=$DG k=$KGRID" >&2
    failures=$((failures + 1))
    continue
  fi

  if (( HOLDOUT == 0 )); then
    QPOINTS=("${QPOINTS_DEVELOPMENT[@]}")
  elif [[ "$KGRID" == 64 ]]; then
    QPOINTS=("${QPOINTS_K64[@]}")
  else
    QPOINTS=("${QPOINTS_HOLDOUT[@]}")
  fi

  for qt in "${QPOINTS[@]}"; do
    REGION="${qt%%:*}"
    T="${qt##*:}"
    SLUG="${REGION}_t${T//./p}"
    QDIR="$SETDIR/$SLUG"
    mkdir -p "$QDIR"
    read -r QX QY < <(awk -v t="$T" 'BEGIN {printf "%.10f %.10f\n", t/3.0, t/sqrt(3.0)}')
    cat > "$QDIR/ph.in" <<EOF
graphene q-space campaign $CAMPAIGN, dg=$DG, k=$KGRID, $REGION t=$T
&inputph
 prefix='gr', outdir='../tmp', fildyn='gr.dyn',
 tr2_ph=1.0d-14, ldisp=.false., trans=.true., epsil=.false., recover=.false.
/
$QX $QY 0.0
EOF
    if grep -q "JOB DONE" "$QDIR/ph.out" 2>/dev/null; then
      echo "[$CAMPAIGN] reuse dg=$DG k=$KGRID $REGION t=$T"
      continue
    fi
    echo "[$CAMPAIGN] PH dg=$DG k=$KGRID $REGION t=$T q=($QX,$QY) $(date -Is)"
    if ! (
      cd "$QDIR"
      nice -n 10 "$QEBIN/mpirun" --allow-run-as-root -np "$NP" \
        "$QEBIN/ph.x" -in ph.in > ph.out 2>&1
    ) || ! grep -q "JOB DONE" "$QDIR/ph.out"; then
      echo "[$CAMPAIGN] PH FAILED dg=$DG k=$KGRID $REGION t=$T" >&2
      failures=$((failures + 1))
    fi
  done
done

CSV="$WORK/dfpt_${CAMPAIGN}.csv"
"$CONDA" run --no-capture-output -n phonon python - \
  "$CAMPAIGN" "$WORK" "$CSV" "$EXPECTED_ROWS" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path

campaign, work, csv_path, expected = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
rows = []
for setdir in sorted(work.glob("dg*_k*")):
    match = re.fullmatch(r"dg(.+)_k(\d+)", setdir.name)
    if not match:
        continue
    dg, kgrid = float(match.group(1)), int(match.group(2))
    for qdir in sorted(setdir.glob("[GK]_t*")):
        out, inp = qdir / "ph.out", qdir / "ph.in"
        if not out.is_file() or "JOB DONE" not in out.read_text(errors="ignore"):
            continue
        qmatch = re.search(
            r"\n\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$",
            inp.read_text(),
        )
        tmatch = re.fullmatch(r"([GK])_t(.+)", qdir.name)
        if not qmatch or not tmatch:
            continue
        frequencies = [
            float(value)
            for value in re.findall(
                r"^\s*freq\s*\(\s*\d+\s*\)\s*=.*?=\s*([-+0-9.Ee]+)\s*\[cm-1\]",
                out.read_text(errors="ignore"),
                flags=re.MULTILINE,
            )
        ]
        if len(frequencies) != 6 or not all(math.isfinite(value) for value in frequencies):
            continue
        rows.append(
            [
                campaign,
                dg,
                kgrid,
                tmatch.group(1),
                float(tmatch.group(2).replace("p", ".")),
                float(qmatch.group(1)),
                float(qmatch.group(2)),
                *frequencies,
            ]
        )
rows.sort(key=lambda row: (row[1], row[2], row[3], row[4]))
if len(rows) != expected:
    raise SystemExit(f"expected {expected} completed q points, found {len(rows)}")
with csv_path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        ["campaign", "degauss_Ry", "kgrid", "region", "t_GK", "qx_2pia", "qy_2pia"]
        + [f"f{i}_cm" for i in range(1, 7)]
    )
    writer.writerows(rows)
print(f"VALID_{campaign}_DONE rows={len(rows)} csv={csv_path}")
PY

if (( failures == 0 )); then
  touch "$WORK/DONE"
  echo "=== graphene q-space DFPT campaign=$CAMPAIGN COMPLETE $(date -Is) ==="
else
  echo "=== campaign=$CAMPAIGN ended with $failures failures $(date -Is) ===" >&2
  exit 1
fi

