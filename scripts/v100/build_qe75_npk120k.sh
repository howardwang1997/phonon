#!/usr/bin/env bash
# Build a separate Quantum ESPRESSO 7.5 tree with a larger k-point ceiling.
# The existing conda QE is never modified.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
WORK_ROOT="${WORK_ROOT:-/data/graphene_k_cusp_nosmear/qe_build}"
BUILD="${BUILD:-/data/qe-7.5-npk120k}"
ARCHIVE="${ARCHIVE:-$WORK_ROOT/q-e-qe-7.5.tar.gz}"
URL="${URL:-https://gitlab.com/QEF/q-e/-/archive/qe-7.5/q-e-qe-7.5.tar.gz}"
NPK="${NPK:-120000}"
JOBS="${JOBS:-8}"

mkdir -p "$WORK_ROOT"
exec 9>"$WORK_ROOT/.lock"
if ! flock -n 9; then
    echo "[qe75-npk] another build is running"
    exit 0
fi
if [[ -s "$WORK_ROOT/DONE" ]]; then
    test -x "$BUILD/bin/pw.x"
    test -x "$BUILD/bin/ph.x"
    echo "[qe75-npk] already complete"
    exit 0
fi

LOG="$WORK_ROOT/build.log"
exec > >(tee -a "$LOG") 2>&1
echo "[qe75-npk] start=$(date -Is) build=$BUILD npk=$NPK jobs=$JOBS"

if [[ ! -s "$ARCHIVE" ]]; then
    echo "[qe75-npk] download $URL"
    curl -fL --retry 4 --retry-delay 10 "$URL" -o "$ARCHIVE.partial"
    mv "$ARCHIVE.partial" "$ARCHIVE"
fi

if [[ ! -d "$BUILD" ]]; then
    extract_root="$WORK_ROOT/extract.$(date +%s)"
    mkdir -p "$extract_root"
    tar -xzf "$ARCHIVE" -C "$extract_root"
    extracted="$(find "$extract_root" -mindepth 1 -maxdepth 1 -type d | head -1)"
    test -n "$extracted"
    mv "$extracted" "$BUILD"
    rmdir "$extract_root"
fi

PARAMETERS="$BUILD/Modules/parameters.f90"
test -s "$PARAMETERS"
if ! grep -Eq "npk[[:space:]]*=[[:space:]]*$NPK([[:space:]]|$)" "$PARAMETERS"; then
    sed -i.npk40000 -E \
        "s/(INTEGER,[[:space:]]*PARAMETER[[:space:]]*::[[:space:]]*npk[[:space:]]*=[[:space:]]*)[0-9]+/\\1$NPK/" \
        "$PARAMETERS"
fi
grep -nE "npk[[:space:]]*=[[:space:]]*$NPK([[:space:]]|$)" "$PARAMETERS"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /root/nvhpc_env.sh
cd "$BUILD"
if [[ ! -s make.inc ]]; then
    echo "[qe75-npk] configure start=$(date -Is)"
    ./configure \
        --with-cuda=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda \
        --with-cuda-runtime=12.4 \
        --with-cuda-cc=70 \
        --enable-openmp \
        MPIF90=mpif90 F90=nvfortran CC=pgcc \
        > "$WORK_ROOT/configure.log" 2>&1
fi

# V100 VMs may move between AVX-512 and AVX2 hosts after reboot.
if ! grep -q -- '-tp=x86-64-v3' make.inc; then
    sed -i.qe-x86-v3-backup -E \
        '/^(CFLAGS|F90FLAGS|FFLAGS|FOX_FLAGS)[[:space:]]*=/ s/$/ -tp=x86-64-v3/' \
        make.inc
fi

echo "[qe75-npk] compile start=$(date -Is)"
make -j"$JOBS" pw ph > "$WORK_ROOT/make.log" 2>&1
test -x "$BUILD/bin/pw.x"
test -x "$BUILD/bin/ph.x"

# NVHPC MPI executables must be launched through the matching mpirun.  A
# successful link alone does not detect a broken MPI runtime prefix.
export OMP_NUM_THREADS=1
SMOKE_LOG="$WORK_ROOT/runtime_smoke.log"
set +e
mpirun --allow-run-as-root -np 1 "$BUILD/bin/pw.x" -in /dev/null \
    > "$SMOKE_LOG" 2>&1
pw_smoke_status=$?
mpirun --allow-run-as-root -np 1 "$BUILD/bin/ph.x" -in /dev/null \
    >> "$SMOKE_LOG" 2>&1
ph_smoke_status=$?
set -e
grep -q "Program PWSCF v.7.5 starts" "$SMOKE_LOG"
grep -q "Program PHONON v.7.5 starts" "$SMOKE_LOG"
if grep -Eq "MPI_Init(_thread)?|opal_init:startup:internal-failure" "$SMOKE_LOG"; then
    echo "[qe75-npk] MPI runtime smoke failed" >&2
    exit 7
fi
printf 'pw_empty_input_exit=%s ph_empty_input_exit=%s\n' \
    "$pw_smoke_status" "$ph_smoke_status" >> "$SMOKE_LOG"

"$CONDA" run --no-capture-output -n phonon python - \
    "$BUILD" "$WORK_ROOT" "$ARCHIVE" "$SMOKE_LOG" "$NPK" <<'PY'
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

build, work, archive, smoke_log = map(Path, sys.argv[1:5])
expected_npk = int(sys.argv[5])


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


parameters = build / "Modules" / "parameters.f90"
match = re.search(r"\bnpk\s*=\s*(\d+)", parameters.read_text())
if match is None or int(match.group(1)) != expected_npk:
    raise RuntimeError("npk audit failed")
payload = {
    "status": "complete",
    "completed_at_iso": datetime.now().astimezone().isoformat(),
    "quantum_espresso_version": "7.5",
    "npk": expected_npk,
    "build": str(build),
    "source_archive": {"path": str(archive), "sha256": digest(archive)},
    "parameters_f90_sha256": digest(parameters),
    "pw": {"path": str(build / "bin" / "pw.x"), "sha256": digest(build / "bin" / "pw.x")},
    "ph": {"path": str(build / "bin" / "ph.x"), "sha256": digest(build / "bin" / "ph.x")},
    "runtime_smoke": {"path": str(smoke_log), "sha256": digest(smoke_log)},
    "make_inc_sha256": digest(build / "make.inc"),
}
temporary = work / "build_audit.json.tmp"
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(work / "build_audit.json")
print(json.dumps(payload, indent=2))
PY

date -Is > "$WORK_ROOT/COMPLETED_AT"
cp "$WORK_ROOT/COMPLETED_AT" "$WORK_ROOT/DONE.tmp"
mv "$WORK_ROOT/DONE.tmp" "$WORK_ROOT/DONE"
echo "[qe75-npk] complete=$(cat "$WORK_ROOT/COMPLETED_AT")"
