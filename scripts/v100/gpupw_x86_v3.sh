#!/usr/bin/env bash
# Portable V100 QE launcher for VMs that may move between AVX2 and AVX-512 CPUs.
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /root/nvhpc_env.sh
exec mpirun --allow-run-as-root -np 1 \
  /data/q-e-qe-7.3.1-x86-v3/bin/pw.x "$@"
