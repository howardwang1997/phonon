#!/usr/bin/env bash
# Restartable direct-DFPT convergence and line-cut campaigns for graphene at
# physical Fermi-Dirac broadenings.  The convergence k meshes are all
# multiples of three so that the graphene Dirac K point is sampled exactly.
set -euo pipefail

CAMPAIGN="${1:?usage: bash scripts/v100/run_graphene_physical_fd_dfpt.sh FD300_CONV|FD600_CONV|FD450_CONV|FD300_K144|FD600_K144|FD300_LINE|FD450_LINE|FD600_LINE|FD0_SCAN}"

QPOINTS_CONV=(
  "G:0.000" "G:0.015" "G:0.030"
  "K:0.980" "K:0.990" "K:1.000" "K:1.010" "K:1.020"
)
QPOINTS_LINE=(
  "G:0.000" "G:0.005" "G:0.010" "G:0.015" "G:0.025"
  "G:0.040" "G:0.060" "G:0.080"
  "K:0.940" "K:0.960" "K:0.975" "K:0.985" "K:0.992"
  "K:1.000" "K:1.008" "K:1.015" "K:1.025" "K:1.040" "K:1.060"
)
QPOINTS_ZERO=(
  "G:0.000" "G:0.015"
  "K:0.985" "K:1.000" "K:1.015"
)

case "$CAMPAIGN" in
  FD300_CONV)
    SETS=("0.0019000869:72" "0.0019000869:96" "0.0019000869:120")
    QPOINTS=("${QPOINTS_CONV[@]}")
    ;;
  FD600_CONV)
    SETS=("0.0038001738:72" "0.0038001738:96" "0.0038001738:120")
    QPOINTS=("${QPOINTS_CONV[@]}")
    ;;
  FD450_CONV)
    SETS=("0.00285013035:120" "0.00285013035:144")
    QPOINTS=("${QPOINTS_CONV[@]}")
    ;;
  FD300_K144)
    SETS=("0.0019000869:144")
    QPOINTS=("${QPOINTS_CONV[@]}")
    ;;
  FD600_K144)
    SETS=("0.0038001738:144")
    QPOINTS=("${QPOINTS_CONV[@]}")
    ;;
  FD300_LINE)
    SETS=("0.0019000869:${KGRID_OVERRIDE:-144}")
    QPOINTS=("${QPOINTS_LINE[@]}")
    ;;
  FD600_LINE)
    SETS=("0.0038001738:${KGRID_OVERRIDE:-120}")
    QPOINTS=("${QPOINTS_LINE[@]}")
    ;;
  FD450_LINE)
    SETS=("0.00285013035:${KGRID_OVERRIDE:-144}")
    QPOINTS=("${QPOINTS_LINE[@]}")
    ;;
  FD0_SCAN)
    # A literal zero-width FD calculation is undefined numerically.  These
    # small-width points support an explicit degauss -> 0 extrapolation.
    SETS=("0.0006333623:192" "0.0012667246:144" "0.0019000869:120")
    QPOINTS=("${QPOINTS_ZERO[@]}")
    ;;
  *)
    echo "unknown campaign '$CAMPAIGN'" >&2
    exit 2
    ;;
esac

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
PSEUDO="${PSEUDO:-$ROOT/pseudo}"
WORK_BASE="${WORK_BASE:-/data/graphene_physical_fd_dfpt}"
WORK="$WORK_BASE/campaigns/$CAMPAIGN"
SET_ROOT="$WORK_BASE/sets"
NP="${NP:-8}"
QEBIN="${QEBIN:-/root/miniconda3/envs/qe/bin}"

if [[ ! -x "$QEBIN/pw.x" || ! -x "$QEBIN/ph.x" ]]; then
  for candidate in \
    /root/miniconda3/envs/qe/bin \
    /root/miniconda3/envs/phonon/bin \
    /root/qe/bin \
    /root/q-e/bin \
    /root/q-e-qe-7.4/bin \
    /opt/qe/bin; do
    if [[ -x "$candidate/pw.x" && -x "$candidate/ph.x" ]]; then
      QEBIN="$candidate"
      break
    fi
  done
fi

MPIRUN="$QEBIN/mpirun"
if [[ ! -x "$MPIRUN" ]]; then
  MPIRUN="$(command -v mpirun || true)"
fi
for executable in "$QEBIN/pw.x" "$QEBIN/ph.x" "$MPIRUN" "$CONDA"; do
  if [[ -z "$executable" || ! -x "$executable" ]]; then
    echo "missing executable: $executable" >&2
    exit 2
  fi
done
if [[ "$CAMPAIGN" == FD450_CONV || "$CAMPAIGN" == FD450_LINE ]]; then
  FREEZE="${FREEZE:-$ROOT/results/graphene_fd_transferability/freeze_manifest.json}"
  if [[ ! -s "$FREEZE" ]]; then
    echo "refusing $CAMPAIGN: missing frozen 450 K predictor at $FREEZE" >&2
    exit 2
  fi
  "$CONDA" run -n phonon python -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "frozen_before_450_holdout"
assert p["direct_450_targets_read"] is False
t=p["T450_on_policy"]
assert t["temperature_K"] == 450
assert abs(t["degauss_Ry"] - 0.00285013035) < 1e-12
' "$FREEZE"
fi
if [[ ! -s "$PSEUDO/C_ONCV_PBE-1.2.upf" ]]; then
  echo "missing pseudopotential: $PSEUDO/C_ONCV_PBE-1.2.upf" >&2
  exit 2
fi

mkdir -p "$WORK" "$SET_ROOT"
exec 9>"$WORK/.campaign.lock"
if ! flock -n 9; then
  echo "$CAMPAIGN is already running; refusing a duplicate" >&2
  exit 0
fi
if [[ -e "$WORK/DONE" ]]; then
  echo "$CAMPAIGN already complete"
  exit 0
fi

LOG="$WORK/run.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== graphene physical-FD DFPT campaign=$CAMPAIGN start $(date -Is) ==="
echo "QEBIN=$QEBIN MPIRUN=$MPIRUN NP=$NP WORK=$WORK SET_ROOT=$SET_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Exact lattice used by the existing 8x8 finite-displacement and direct-DFPT
# reference data.  Keeping it fixed isolates electronic-smearing and k-mesh
# convergence from structural changes.
A_BOHR=4.648726286
C_OVER_A=6.097560976

run_scf() {
  local setdir="$1"
  if grep -q "JOB DONE" "$setdir/scf.out" 2>/dev/null; then
    echo "[$CAMPAIGN] reuse $(basename "$setdir") SCF"
    return 0
  fi
  if [[ -s "$setdir/scf.out" ]]; then
    cp -p "$setdir/scf.out" "$setdir/scf.out.failed.$(date +%s)"
  fi
  echo "[$CAMPAIGN] SCF $(basename "$setdir") $(date -Is)"
  (
    cd "$setdir"
    "$MPIRUN" --allow-run-as-root -np "$NP" \
      "$QEBIN/pw.x" -in scf.in > scf.out 2>&1
  )
  grep -q "JOB DONE" "$setdir/scf.out"
}

run_ph() {
  local qdir="$1"
  local attempt
  if grep -q "JOB DONE" "$qdir/ph.out" 2>/dev/null; then
    return 0
  fi
  for attempt in 1 2; do
    if [[ -s "$qdir/ph.out" ]]; then
      cp -p "$qdir/ph.out" "$qdir/ph.out.failed.$(date +%s).attempt$attempt"
    fi
    echo "[$CAMPAIGN] PH attempt=$attempt $(basename "$(dirname "$qdir")")/$(basename "$qdir") $(date -Is)"
    if (
      cd "$qdir"
      "$MPIRUN" --allow-run-as-root -np "$NP" \
        "$QEBIN/ph.x" -in ph.in > ph.out 2>&1
    ) && grep -q "JOB DONE" "$qdir/ph.out"; then
      return 0
    fi
  done
  return 1
}

failures=0
for spec in "${SETS[@]}"; do
  DG="${spec%%:*}"
  KGRID="${spec##*:}"
  # Force calculations are cached by physical setting, not by campaign.  A
  # later dense line cut therefore reuses the SCF and any already-computed q
  # points from the convergence stage.
  SETDIR="$SET_ROOT/dg${DG}_k${KGRID}"
  mkdir -p "$SETDIR/tmp"

  cat > "$SETDIR/scf.in" <<EOF
&control
 calculation='scf', prefix='gr', outdir='./tmp', pseudo_dir='$PSEUDO',
 verbosity='low', disk_io='low'
/
&system
 ibrav=4, celldm(1)=$A_BOHR, celldm(3)=$C_OVER_A, nat=2, ntyp=1,
 ecutwfc=60, ecutrho=240,
 occupations='smearing', smearing='fd', degauss=$DG
/
&electrons
 conv_thr=1.0d-12, mixing_beta=0.3, electron_maxstep=300,
 diagonalization='david'
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

  for qt in "${QPOINTS[@]}"; do
    REGION="${qt%%:*}"
    T="${qt##*:}"
    SLUG="${REGION}_t${T//./p}"
    QDIR="$SETDIR/$SLUG"
    mkdir -p "$QDIR"
    read -r QX QY < <(awk -v t="$T" 'BEGIN {printf "%.12f %.12f\n", t/3.0, t/sqrt(3.0)}')
    cat > "$QDIR/ph.in" <<EOF
graphene physical-FD DFPT, campaign=$CAMPAIGN, degauss=$DG Ry, k=$KGRID, $REGION t=$T
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
    if ! run_ph "$QDIR"; then
      echo "[$CAMPAIGN] PH FAILED dg=$DG k=$KGRID $REGION t=$T" >&2
      failures=$((failures + 1))
    fi
  done
done

CSV="$WORK/graphene_${CAMPAIGN}_dfpt.csv"
EXPECTED_ROWS=$((${#SETS[@]} * ${#QPOINTS[@]}))
SET_SPECS=$(IFS=,; echo "${SETS[*]}")
Q_SPECS=$(IFS=,; echo "${QPOINTS[*]}")
"$CONDA" run --no-capture-output -n phonon python - \
  "$CAMPAIGN" "$SET_ROOT" "$CSV" "$EXPECTED_ROWS" "$SET_SPECS" "$Q_SPECS" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path

campaign = sys.argv[1]
set_root = Path(sys.argv[2])
csv_path = Path(sys.argv[3])
expected = int(sys.argv[4])
set_specs = sys.argv[5].split(",")
q_specs = sys.argv[6].split(",")
rows = []
allowed_sets = {f"dg{spec.split(':')[0]}_k{spec.split(':')[1]}" for spec in set_specs}
allowed_qdirs = {
    f"{spec.split(':')[0]}_t{spec.split(':')[1].replace('.', 'p')}"
    for spec in q_specs
}
for setdir in sorted(set_root.glob("dg*_k*")):
    if setdir.name not in allowed_sets:
        continue
    match = re.fullmatch(r"dg(.+)_k(\d+)", setdir.name)
    if not match:
        continue
    degauss = float(match.group(1))
    kgrid = int(match.group(2))
    for qdir in sorted(setdir.glob("[GK]_t*")):
        if qdir.name not in allowed_qdirs:
            continue
        output = qdir / "ph.out"
        input_path = qdir / "ph.in"
        if not output.is_file() or "JOB DONE" not in output.read_text(errors="ignore"):
            continue
        tmatch = re.fullmatch(r"([GK])_t(.+)", qdir.name)
        qmatch = re.search(
            r"\n\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$",
            input_path.read_text(),
        )
        frequencies = [
            float(value)
            for value in re.findall(
                r"^\s*freq\s*\(\s*\d+\s*\)\s*=.*?=\s*([-+0-9.Ee]+)\s*\[cm-1\]",
                output.read_text(errors="ignore"),
                flags=re.MULTILINE,
            )
        ]
        if not tmatch or not qmatch or len(frequencies) != 6:
            continue
        if not all(math.isfinite(value) for value in frequencies):
            continue
        rows.append(
            [
                campaign,
                degauss,
                kgrid,
                tmatch.group(1),
                float(tmatch.group(2).replace("p", ".")),
                float(qmatch.group(1)),
                float(qmatch.group(2)),
                *frequencies,
            ]
        )

rows.sort(key=lambda row: (row[1], row[2], row[3], row[4]))
with csv_path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        ["campaign", "degauss_Ry", "kgrid", "region", "t_GK", "qx_2pia", "qy_2pia"]
        + [f"f{index}_cm-1" for index in range(1, 7)]
    )
    writer.writerows(rows)
print(f"wrote {csv_path} rows={len(rows)}/{expected}")
if len(rows) != expected:
    raise SystemExit(f"expected {expected} completed q points, found {len(rows)}")
PY

if (( failures == 0 )); then
  touch "$WORK/DONE"
  echo "=== graphene physical-FD DFPT campaign=$CAMPAIGN COMPLETE $(date -Is) ==="
else
  echo "=== campaign=$CAMPAIGN ended with $failures failures $(date -Is) ===" >&2
  exit 1
fi
