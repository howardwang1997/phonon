#!/usr/bin/env bash
# Rebuild the NVHPC/CUDA Quantum ESPRESSO binary for an AVX2-compatible CPU.
#
# V100 virtual machines can move between Cascade Lake (AVX-512) and Broadwell
# (AVX2) hosts after a reboot.  A binary built with NVHPC's implicit host
# target can then terminate with SIGILL before parsing the QE input.  Keep the
# original tree untouched and build an x86-64-v3 copy on /data instead.
set -euo pipefail

SOURCE="${SOURCE:-/root/q-e-qe-7.3.1}"
BUILD="${BUILD:-/data/q-e-qe-7.3.1-x86-v3}"
JOBS="${JOBS:-8}"

if [[ ! -f "$SOURCE/make.inc" ]]; then
  echo "missing configured QE source: $SOURCE" >&2
  exit 2
fi

if [[ ! -d "$BUILD" ]]; then
  cp -a "$SOURCE" "$BUILD"
fi

cd "$BUILD"
if ! grep -q -- '-tp=x86-64-v3' make.inc; then
  sed -i.qe-x86-v3-backup -E \
    '/^(CFLAGS|F90FLAGS|FFLAGS|FOX_FLAGS)[[:space:]]*=/ s/$/ -tp=x86-64-v3/' \
    make.inc
fi

# nvhpc_env.sh appends to this variable and assumes it already exists.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /root/nvhpc_env.sh
make clean
make -j"$JOBS" pw

test -x "$BUILD/bin/pw.x"
if objdump -d "$BUILD/bin/pw.x" | grep -Eq '\{%(k[0-7])\}|%k[0-7]'; then
  echo "warning: mask-register instructions remain in $BUILD/bin/pw.x" >&2
fi
echo "built $BUILD/bin/pw.x"
