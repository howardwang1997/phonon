#!/usr/bin/env bash
# Add symmetric k192 DFPT points around K at the lowest finite degauss.
set -euo pipefail

LANE="${1:?usage: run_graphene_k_cusp_c0_dfpt_dense.sh A|B}"
case "$LANE" in
    A)
        QPOINTS=(0.977 0.981 0.989 0.993 0.997 1.000)
        ;;
    B)
        QPOINTS=(1.003 1.007 1.011 1.019 1.023)
        ;;
    *)
        echo "unknown lane: $LANE" >&2
        exit 2
        ;;
esac

if [[ -n "${QPOINTS_OVERRIDE:-}" ]]; then
    IFS=',' read -r -a QPOINTS <<< "$QPOINTS_OVERRIDE"
fi

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
PSEUDO="${PSEUDO:-$ROOT/pseudo}"
WORK_BASE="${WORK_BASE:-/data/graphene_physical_fd_dfpt}"
SET_ROOT="$WORK_BASE/sets"
DG="${DG:-0.0006333623}"
KGRID="${KGRID:-192}"
SETDIR="$SET_ROOT/dg${DG}_k${KGRID}"
WORK="${WORK:-/data/graphene_k_cusp_c0/FD0_K_DENSE_${LANE}}"
NP="${NP:-8}"
QEBIN="${QEBIN:-/root/miniconda3/envs/qe/bin}"
WAIT_FOR="${WAIT_FOR:-}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-21600}"

if [[ -n "$WAIT_FOR" ]]; then
    waited=0
    echo "[cusp-c0-$LANE] waiting for $WAIT_FOR"
    while [[ ! -s "$WAIT_FOR" ]]; do
        sleep 60
        waited=$((waited + 60))
        if (( waited >= WAIT_TIMEOUT_SECONDS )); then
            echo "[cusp-c0-$LANE] wait timeout after ${waited}s" >&2
            exit 3
        fi
    done
fi

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
test -s "$PSEUDO/C_ONCV_PBE-1.2.upf"

mkdir -p "$WORK" "$SETDIR/tmp"
exec 9>"$WORK/.lock"
if ! flock -n 9; then
    echo "[cusp-c0-$LANE] another process is running"
    exit 0
fi
if [[ -s "$WORK/DONE" ]]; then
    echo "[cusp-c0-$LANE] already complete"
    exit 0
fi
LOG="$WORK/run.log"
exec > >(tee -a "$LOG") 2>&1
echo "[cusp-c0-$LANE] start=$(date -Is) DG=$DG k=$KGRID NP=$NP"
echo "[cusp-c0-$LANE] QEBIN=$QEBIN MPIRUN=$MPIRUN PSEUDO=$PSEUDO"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cat > "$SETDIR/scf.in" <<EOF
&control
 calculation='scf', prefix='gr', outdir='./tmp', pseudo_dir='$PSEUDO',
 verbosity='low', disk_io='low'
/
&system
 ibrav=4, celldm(1)=4.648726286, celldm(3)=6.097560976, nat=2, ntyp=1,
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

if grep -q "JOB DONE" "$SETDIR/scf.out" 2>/dev/null; then
    echo "[cusp-c0-$LANE] reuse SCF"
else
    if [[ -s "$SETDIR/scf.out" ]]; then
        cp -p "$SETDIR/scf.out" "$SETDIR/scf.out.failed.$(date +%s)"
    fi
    echo "[cusp-c0-$LANE] SCF start=$(date -Is)"
    (
        cd "$SETDIR"
        "$MPIRUN" --allow-run-as-root -np "$NP" \
            "$QEBIN/pw.x" -in scf.in > scf.out 2>&1
    )
    grep -q "JOB DONE" "$SETDIR/scf.out"
    echo "[cusp-c0-$LANE] SCF complete=$(date -Is)"
fi

run_ph() {
    local t="$1"
    local slug="K_t${t//./p}"
    local qdir="$SETDIR/$slug"
    local qx qy attempt
    mkdir -p "$qdir"
    read -r qx qy < <(awk -v value="$t" 'BEGIN {printf "%.12f %.12f\n", value/3.0, value/sqrt(3.0)}')
    cat > "$qdir/ph.in" <<EOF
graphene C0 dense K DFPT, lane=$LANE, degauss=$DG Ry, k=$KGRID, t=$t
&inputph
 prefix='gr', outdir='../tmp', fildyn='gr.dyn',
 tr2_ph=1.0d-14, ldisp=.false., trans=.true., epsil=.false., recover=.false.
/
$qx $qy 0.0
EOF
    if grep -q "JOB DONE" "$qdir/ph.out" 2>/dev/null; then
        echo "[cusp-c0-$LANE] reuse K t=$t"
        return 0
    fi
    for attempt in 1 2; do
        if [[ -s "$qdir/ph.out" ]]; then
            cp -p "$qdir/ph.out" "$qdir/ph.out.failed.$(date +%s).attempt$attempt"
        fi
        echo "[cusp-c0-$LANE] PH t=$t attempt=$attempt start=$(date -Is)"
        if (
            cd "$qdir"
            "$MPIRUN" --allow-run-as-root -np "$NP" \
                "$QEBIN/ph.x" -in ph.in > ph.out 2>&1
        ) && grep -q "JOB DONE" "$qdir/ph.out"; then
            echo "[cusp-c0-$LANE] PH t=$t complete=$(date -Is)"
            return 0
        fi
    done
    return 1
}

failures=0
for t in "${QPOINTS[@]}"; do
    if ! run_ph "$t"; then
        echo "[cusp-c0-$LANE] PH FAILED t=$t" >&2
        failures=$((failures + 1))
    fi
done
if (( failures != 0 )); then
    echo "[cusp-c0-$LANE] stopped with $failures failed q points" >&2
    exit 1
fi

Q_SPECS="$(IFS=,; echo "${QPOINTS[*]}")"
CSV="$WORK/graphene_FD0_K_DENSE_${LANE}_dfpt.csv"
"$CONDA" run --no-capture-output -n phonon python - \
    "$SETDIR" "$CSV" "$Q_SPECS" "$DG" "$KGRID" "$LANE" \
    "$PSEUDO/C_ONCV_PBE-1.2.upf" <<'PY'
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

setdir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
t_values = [float(value) for value in sys.argv[3].split(",")]
degauss = float(sys.argv[4])
kgrid = int(sys.argv[5])
lane = sys.argv[6]
pseudo_path = Path(sys.argv[7])
rows = []
for t_value in t_values:
    qdir = setdir / f"K_t{t_value:.3f}".replace(".", "p")
    output = qdir / "ph.out"
    input_path = qdir / "ph.in"
    text = output.read_text(errors="replace")
    qmatch = re.search(
        r"\n\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$",
        input_path.read_text(),
    )
    frequencies = [
        float(value)
        for value in re.findall(
            r"^\s*freq\s*\(\s*\d+\s*\)\s*=.*?=\s*([-+0-9.Ee]+)\s*\[cm-1\]",
            text,
            flags=re.MULTILINE,
        )
    ]
    if "JOB DONE" not in text or qmatch is None or len(frequencies) != 6:
        raise RuntimeError(f"incomplete DFPT output at t={t_value:.3f}")
    if not all(math.isfinite(value) for value in frequencies):
        raise RuntimeError(f"non-finite DFPT frequency at t={t_value:.3f}")
    rows.append(
        [
            f"FD0_K_DENSE_{lane}", degauss, kgrid, "K", t_value,
            float(qmatch.group(1)), float(qmatch.group(2)), *frequencies,
        ]
    )

with csv_path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        ["campaign", "degauss_Ry", "kgrid", "region", "t_GK", "qx_2pia", "qy_2pia"]
        + [f"f{index}_cm-1" for index in range(1, 7)]
    )
    writer.writerows(rows)

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

manifest = {
    "status": "complete",
    "lane": lane,
    "degauss_Ry": degauss,
    "kgrid": kgrid,
    "t_values": t_values,
    "n_qpoints": len(rows),
    "csv": {"path": str(csv_path), "sha256": digest(csv_path)},
    "scf_input_sha256": digest(setdir / "scf.in"),
    "scf_output_sha256": digest(setdir / "scf.out"),
    "pseudopotential": {"path": str(pseudo_path), "sha256": digest(pseudo_path)},
}
(csv_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

date -Is > "$WORK/COMPLETED_AT"
cp "$WORK/COMPLETED_AT" "$WORK/DONE.tmp"
mv "$WORK/DONE.tmp" "$WORK/DONE"
echo "[cusp-c0-$LANE] complete=$(cat "$WORK/COMPLETED_AT")"
