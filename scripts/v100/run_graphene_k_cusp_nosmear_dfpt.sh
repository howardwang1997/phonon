#!/usr/bin/env bash
# Direct graphene DFPT with optimized-tetrahedron electronic integration.
#
# Usage:
#   run_graphene_k_cusp_nosmear_dfpt.sh A|B pilot
#   KGRID=240 QE_TAG=qe75_npk120k QEBIN=/data/qe-7.5-npk120k/bin \
#     run_graphene_k_cusp_nosmear_dfpt.sh A|B convergence
#   CONFIG=.../qpoints_d003_kgrid.tsv KGRID=240 \
#     run_graphene_k_cusp_nosmear_dfpt.sh A|B diagnostic
#
# Development and holdout stages are release-gated.  Every completed lane is
# self-contained: inputs, outputs and gr.dyn files are copied into its bundle.
set -euo pipefail

LANE="${1:?usage: run_graphene_k_cusp_nosmear_dfpt.sh A|B STAGE}"
STAGE="${2:?usage: run_graphene_k_cusp_nosmear_dfpt.sh A|B STAGE}"
case "$LANE" in A|B) ;; *) echo "invalid lane: $LANE" >&2; exit 2 ;; esac
case "$STAGE" in pilot|development|holdout|convergence|diagnostic) ;; *) echo "invalid stage: $STAGE" >&2; exit 2 ;; esac

ROOT="${ROOT:-/root/phonon}"
CONFIG="${CONFIG:-$ROOT/configs/graphene_k_cusp_nosmear/qpoints.tsv}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
PSEUDO="${PSEUDO:-$ROOT/pseudo}"
WORK_ROOT="${WORK_ROOT:-/data/graphene_k_cusp_nosmear}"
KGRID="${KGRID:-192}"
QE_TAG="${QE_TAG:-qe75_conda}"
QEBIN="${QEBIN:-/root/miniconda3/envs/phonon/bin}"
QE_ENV_SCRIPT="${QE_ENV_SCRIPT:-}"
NP="${NP:-8}"
OMP_THREADS="${OMP_THREADS:-1}"
MIN_FREE_GIB="${MIN_FREE_GIB:-35}"
RECOVER_LABEL="${RECOVER_LABEL:-}"

test -s "$CONFIG"
test -s "$PSEUDO/C_ONCV_PBE-1.2.upf"
if [[ -n "$QE_ENV_SCRIPT" ]]; then
    test -s "$QE_ENV_SCRIPT"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    # The independently built NVHPC QE needs its compiler/MPI runtime paths.
    # shellcheck disable=SC1090
    source "$QE_ENV_SCRIPT"
fi
for executable in "$QEBIN/pw.x" "$QEBIN/ph.x" "$CONDA"; do
    test -x "$executable" || { echo "missing executable: $executable" >&2; exit 2; }
done
MPIRUN="${MPIRUN:-$QEBIN/mpirun}"
if [[ ! -x "$MPIRUN" ]]; then
    MPIRUN="$(command -v mpirun || true)"
fi
test -n "$MPIRUN" && test -x "$MPIRUN"

case "$STAGE" in
    convergence)
        test -s "$WORK_ROOT/RELEASE_CONVERGENCE" || {
            echo "convergence is locked: missing $WORK_ROOT/RELEASE_CONVERGENCE" >&2
            exit 4
        }
        ;;
    development)
        test -s "$WORK_ROOT/RELEASE_DEVELOPMENT" || {
            echo "development is locked: missing $WORK_ROOT/RELEASE_DEVELOPMENT" >&2
            exit 4
        }
        ;;
    holdout)
        test -s "$WORK_ROOT/RELEASE_HOLDOUT" || {
            echo "holdout is locked: missing $WORK_ROOT/RELEASE_HOLDOUT" >&2
            exit 4
        }
        selected="$(tr -d '[:space:]' < "$WORK_ROOT/SELECTED_KGRID")"
        [[ "$selected" == "$KGRID" ]] || {
            echo "holdout k=$KGRID does not match frozen k=$selected" >&2
            exit 4
        }
        ;;
    diagnostic)
        test -s "$WORK_ROOT/RELEASE_D003_KGRID" || {
            echo "d003 k-grid diagnostic is locked: missing $WORK_ROOT/RELEASE_D003_KGRID" >&2
            exit 4
        }
        diagnostic_grid="$(tr -d '[:space:]' < "$WORK_ROOT/D003_DIAGNOSTIC_KGRID")"
        [[ "$diagnostic_grid" == "$KGRID" ]] || {
            echo "diagnostic k=$KGRID does not match released k=$diagnostic_grid" >&2
            exit 4
        }
        ;;
esac

free_kib="$(df -Pk "$WORK_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
if [[ -n "$free_kib" ]] && (( free_kib < required_kib )); then
    echo "storage gate failed: free=${free_kib} KiB, required=${required_kib} KiB" >&2
    exit 5
fi

SETDIR="$WORK_ROOT/sets/k${KGRID}_tetra_${QE_TAG}"
LANE_ROOT="$WORK_ROOT/lanes/${STAGE}_${LANE}_k${KGRID}_${QE_TAG}"
BUNDLE="$LANE_ROOT/bundle"
mkdir -p "$SETDIR/tmp" "$LANE_ROOT" "$BUNDLE"
exec 9>"$LANE_ROOT/.lock"
if ! flock -n 9; then
    echo "[nosmear-$STAGE-$LANE] another process is running"
    exit 0
fi
if [[ -s "$LANE_ROOT/DONE" ]]; then
    echo "[nosmear-$STAGE-$LANE] already complete"
    exit 0
fi

LOG="$LANE_ROOT/run.log"
exec > >(tee -a "$LOG") 2>&1
echo "[nosmear-$STAGE-$LANE] start=$(date -Is) k=$KGRID qe_tag=$QE_TAG np=$NP omp=$OMP_THREADS"
echo "[nosmear-$STAGE-$LANE] QEBIN=$QEBIN MPIRUN=$MPIRUN"

mapfile -t POINTS < <(
    awk -F '\t' -v stage="$STAGE" -v lane="$LANE" \
        'NR > 1 && $1 == stage && $2 == lane {print $3 "|" $4 "|" $5 "|" $6}' \
        "$CONFIG"
)
if (( ${#POINTS[@]} == 0 )); then
    echo "no q points for stage=$STAGE lane=$LANE" >&2
    exit 2
fi

export OMP_NUM_THREADS="$OMP_THREADS"
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
 occupations='tetrahedra_opt'
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

if grep -Eiq '(^|[[:space:],])(smearing|degauss)[[:space:]]*=' "$SETDIR/scf.in"; then
    echo "forbidden smearing/degauss keyword in $SETDIR/scf.in" >&2
    exit 6
fi
grep -q "occupations='tetrahedra_opt'" "$SETDIR/scf.in"

input_hash="$(sha256sum "$SETDIR/scf.in" | awk '{print $1}')"
reuse_scf=0
if grep -q "JOB DONE" "$SETDIR/scf.out" 2>/dev/null \
    && [[ "$(cat "$SETDIR/scf.in.sha256" 2>/dev/null || true)" == "$input_hash" ]]; then
    reuse_scf=1
fi
if (( reuse_scf == 1 )); then
    echo "[nosmear-$STAGE-$LANE] reuse SCF"
else
    if [[ -e "$SETDIR/scf.out" ]]; then
        mv "$SETDIR/scf.out" "$SETDIR/scf.out.failed.$(date +%s)"
    fi
    rm -f "$SETDIR/scf.in.sha256"
    echo "[nosmear-$STAGE-$LANE] SCF start=$(date -Is)"
    (
        cd "$SETDIR"
        "$MPIRUN" --allow-run-as-root -np "$NP" \
            "$QEBIN/pw.x" -in scf.in > scf.out 2>&1
    )
    grep -q "JOB DONE" "$SETDIR/scf.out"
    printf '%s\n' "$input_hash" > "$SETDIR/scf.in.sha256"
    echo "[nosmear-$STAGE-$LANE] SCF complete=$(date -Is)"
fi

run_ph() {
    local label="$1" direction="$2" delta="$3" role="$4"
    local qdir="$SETDIR/$label" h k qx qy ph_hash attempt recover_flag
    mkdir -p "$qdir"
    read -r h k < <(
        awk -v direction="$direction" -v d="$delta" 'BEGIN {
            if (direction == "K") {
                h = 1.0 / 3.0; k = 1.0 / 3.0
            } else if (direction == "KG") {
                h = (1.0 - d) / 3.0; k = (1.0 - d) / 3.0
            } else if (direction == "KM") {
                h = (1.0 + d) / 3.0; k = (1.0 - 2.0 * d) / 3.0
            } else {
                exit 2
            }
            printf "%.12f %.12f\n", h, k
        }'
    )
    read -r qx qy < <(
        awk -v h="$h" -v k="$k" 'BEGIN {
            printf "%.12f %.12f\n", h, (h + 2.0 * k) / sqrt(3.0)
        }'
    )
    recover_flag='.false.'
    if [[ -n "$RECOVER_LABEL" && "$label" == "$RECOVER_LABEL" ]]; then
        recover_flag='.true.'
    fi
    cat > "$qdir/ph.in" <<EOF
graphene no-degauss DFPT; stage=$STAGE lane=$LANE label=$label direction=$direction delta=$delta role=$role reduced=($h,$k,0)
&inputph
 prefix='gr', outdir='../tmp', fildyn='gr.dyn',
 tr2_ph=1.0d-14, ldisp=.false., trans=.true., epsil=.false., recover=$recover_flag
/
$qx $qy 0.0
EOF
    ph_hash="$(sha256sum "$qdir/ph.in" | awk '{print $1}')"
    if grep -q "JOB DONE" "$qdir/ph.out" 2>/dev/null \
        && [[ -s "$qdir/gr.dyn" ]] \
        && [[ "$(cat "$qdir/ph.in.sha256" 2>/dev/null || true)" == "$ph_hash" ]]; then
        echo "[nosmear-$STAGE-$LANE] reuse $label"
        return 0
    fi
    for attempt in 1 2; do
        if [[ -e "$qdir/ph.out" ]]; then
            mv "$qdir/ph.out" "$qdir/ph.out.failed.$(date +%s).attempt$attempt"
        fi
        if [[ -e "$qdir/gr.dyn" ]]; then
            mv "$qdir/gr.dyn" "$qdir/gr.dyn.failed.$(date +%s).attempt$attempt"
        fi
        rm -f "$qdir/ph.in.sha256"
        echo "[nosmear-$STAGE-$LANE] PH $label attempt=$attempt start=$(date -Is)"
        if (
            cd "$qdir"
            "$MPIRUN" --allow-run-as-root -np "$NP" \
                "$QEBIN/ph.x" -in ph.in > ph.out 2>&1
        ) && grep -q "JOB DONE" "$qdir/ph.out" && [[ -s "$qdir/gr.dyn" ]]; then
            printf '%s\n' "$ph_hash" > "$qdir/ph.in.sha256"
            echo "[nosmear-$STAGE-$LANE] PH $label complete=$(date -Is)"
            return 0
        fi
    done
    echo "[nosmear-$STAGE-$LANE] PH FAILED $label" >&2
    return 1
}

failures=0
for point in "${POINTS[@]}"; do
    IFS='|' read -r label direction delta role <<< "$point"
    run_ph "$label" "$direction" "$delta" "$role" || failures=$((failures + 1))
done
if (( failures != 0 )); then
    printf 'failed_qpoints=%s\n' "$failures" > "$LANE_ROOT/FAILED"
    exit 1
fi

rm -rf "$BUNDLE.tmp"
mkdir -p "$BUNDLE.tmp/scf"
cp -p "$SETDIR/scf.in" "$SETDIR/scf.out" "$SETDIR/scf.in.sha256" "$BUNDLE.tmp/scf/"
for point in "${POINTS[@]}"; do
    IFS='|' read -r label direction delta role <<< "$point"
    mkdir -p "$BUNDLE.tmp/$label"
    cp -p "$SETDIR/$label/ph.in" "$SETDIR/$label/ph.out" \
        "$SETDIR/$label/ph.in.sha256" "$SETDIR/$label/gr.dyn" \
        "$BUNDLE.tmp/$label/"
done
rm -rf "$BUNDLE"
mv "$BUNDLE.tmp" "$BUNDLE"

POINT_SPECS="$(IFS=,; echo "${POINTS[*]}")"
"$CONDA" run --no-capture-output -n phonon python - \
    "$ROOT" "$CONFIG" "$BUNDLE" "$LANE_ROOT" "$STAGE" "$LANE" \
    "$KGRID" "$QE_TAG" "$QEBIN" "$PSEUDO/C_ONCV_PBE-1.2.upf" \
    "$QE_ENV_SCRIPT" "$NP" "$OMP_THREADS" "$MPIRUN" "$POINT_SPECS" \
    "$RECOVER_LABEL" <<'PY'
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

root, config, bundle, lane_root = map(Path, sys.argv[1:5])
stage, lane = sys.argv[5:7]
kgrid = int(sys.argv[7])
qe_tag = sys.argv[8]
qebin, pseudo = map(Path, sys.argv[9:11])
qe_env_script_text = sys.argv[11]
mpi_ranks = int(sys.argv[12])
omp_threads = int(sys.argv[13])
mpirun_path = Path(sys.argv[14])
point_specs = [item.split("|") for item in sys.argv[15].split(",")]
recover_label = sys.argv[16]
sys.path.insert(0, str(root / "scripts" / "smearing_kink"))
from qe_dyn import load_qe_dyn  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def q_reduced(direction: str, delta: float) -> tuple[float, float]:
    if direction == "K":
        return 1.0 / 3.0, 1.0 / 3.0
    if direction == "KG":
        return (1.0 - delta) / 3.0, (1.0 - delta) / 3.0
    if direction == "KM":
        return (1.0 + delta) / 3.0, (1.0 - 2.0 * delta) / 3.0
    raise ValueError(direction)


scf_input = (bundle / "scf" / "scf.in").read_text()
if "occupations='tetrahedra_opt'" not in scf_input:
    raise RuntimeError("SCF does not use tetrahedra_opt")
if re.search(r"(^|[\s,])(smearing|degauss)\s*=", scf_input, re.I):
    raise RuntimeError("SCF contains forbidden smearing/degauss")
if "JOB DONE" not in (bundle / "scf" / "scf.out").read_text(errors="replace"):
    raise RuntimeError("incomplete SCF")

rows = []
audits = []
for label, direction, delta_text, role in point_specs:
    delta = float(delta_text)
    qdir = bundle / label
    dyn = load_qe_dyn(qdir / "gr.dyn")
    expected_h, expected_k = q_reduced(direction, delta)
    expected_cart = np.asarray(
        [expected_h, (expected_h + 2.0 * expected_k) / math.sqrt(3.0), 0.0]
    )
    q_error = float(np.max(np.abs(dyn.q_cart_2pi_over_a - expected_cart)))
    if q_error > 2.0e-8:
        raise RuntimeError(f"q mismatch for {label}: {q_error}")
    replay = float(
        np.max(np.abs(dyn.frequencies_from_matrix_cm() - dyn.frequencies_cm))
    )
    if dyn.hermitian_error > 5.0e-7 or replay > 0.1:
        raise RuntimeError(
            f"dynamical-matrix audit failed for {label}: "
            f"Hermiticity={dyn.hermitian_error}, replay={replay}"
        )
    row = {
        "stage": stage,
        "lane": lane,
        "kgrid": kgrid,
        "qe_tag": qe_tag,
        "integration": "tetrahedra_opt_no_degauss",
        "label": label,
        "direction": direction,
        "delta_equal_distance": delta,
        "role": role,
        "q_reduced_h": expected_h,
        "q_reduced_k": expected_k,
        "qx_2pia": expected_cart[0],
        "qy_2pia": expected_cart[1],
    }
    row.update({f"f{index}_cm-1": value for index, value in enumerate(dyn.frequencies_cm, 1)})
    rows.append(row)
    audits.append(
        {
            "label": label,
            "gr_dyn_sha256": digest(qdir / "gr.dyn"),
            "ph_in_sha256": digest(qdir / "ph.in"),
            "ph_out_sha256": digest(qdir / "ph.out"),
            "q_max_abs_error": q_error,
            "Hermiticity_max_abs": dyn.hermitian_error,
            "eigenvector_orthogonality_max_abs": dyn.eigenvector_orthogonality_error,
            "matrix_frequency_replay_max_abs_cm-1": replay,
        }
    )

csv_path = lane_root / "dfpt_points.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

manifest = {
    "status": "complete",
    "scope": "graphene K cusp no-degauss direct DFPT stage lane",
    "stage": stage,
    "lane": lane,
    "kgrid": kgrid,
    "qe_tag": qe_tag,
    "electronic_integration": "occupations='tetrahedra_opt'",
    "smearing": None,
    "degauss_Ry": None,
    "parallel": {
        "mpi_ranks": mpi_ranks,
        "openmp_threads_per_rank": omp_threads,
        "mpirun_path": str(mpirun_path),
    },
    "restart": {
        "recover_label": recover_label or None,
        "recovered_from_interrupted_response": bool(recover_label),
    },
    "n_qpoints": len(rows),
    "config": {"path": str(config), "sha256": digest(config)},
    "scf": {
        "input_sha256": digest(bundle / "scf" / "scf.in"),
        "output_sha256": digest(bundle / "scf" / "scf.out"),
    },
    "quantum_espresso": {
        "pw_path": str(qebin / "pw.x"),
        "pw_sha256": digest(qebin / "pw.x"),
        "ph_path": str(qebin / "ph.x"),
        "ph_sha256": digest(qebin / "ph.x"),
        "tag": qe_tag,
        "environment_script": (
            {
                "path": qe_env_script_text,
                "sha256": digest(Path(qe_env_script_text)),
            }
            if qe_env_script_text
            else None
        ),
    },
    "pseudopotential": {"path": str(pseudo), "sha256": digest(pseudo)},
    "qpoint_audits": audits,
    "aggregate_audit": {
        "Hermiticity_max_abs": max(item["Hermiticity_max_abs"] for item in audits),
        "matrix_frequency_replay_max_abs_cm-1": max(
            item["matrix_frequency_replay_max_abs_cm-1"] for item in audits
        ),
    },
    "outputs": {
        "csv": str(csv_path),
        "csv_sha256": digest(csv_path),
        "bundle": str(bundle),
    },
}
temporary = lane_root / "manifest.json.tmp"
temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
temporary.replace(lane_root / "manifest.json")
print(json.dumps(manifest, indent=2))
PY

date -Is > "$LANE_ROOT/COMPLETED_AT"
rm -f "$LANE_ROOT/FAILED"
cp "$LANE_ROOT/COMPLETED_AT" "$LANE_ROOT/DONE.tmp"
mv "$LANE_ROOT/DONE.tmp" "$LANE_ROOT/DONE"
echo "[nosmear-$STAGE-$LANE] complete=$(cat "$LANE_ROOT/COMPLETED_AT")"
