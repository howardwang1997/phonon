# GPU acceleration of phonon-spectrum DFT — engineering report

**Branch:** `glm`
**Host:** AnolisOS NGC container, 8× NVIDIA H20 (97 GB), CUDA driver 12.8
**Date:** 2026-06-18

---

## 1. Objective

DFT phonon-spectrum calculations are far slower than single-point electronic
calculations because they require many supercell SCF force evaluations. This work
builds a GPU pipeline that reduces wall-clock by stacking three independent
mechanisms, and reports **measured** end-to-end speedups.

## 2. The three levers

| # | Lever | Where it acts | This host |
|---|---|---|---|
| 1 | **Symmetry-reduced finite displacements** | number of SCF jobs | 16–64× fewer jobs |
| 2 | **Per-SCF GPU offload** (QE `pw.x` GPU build) | each SCF | 1.5× (16-atom) … 4.75× (64-atom) |
| 3 | **Multi-GPU concurrency** (`MultiGPUExecutor`) | many SCFs at once | ~2× (2× H20) |

They multiply in wall-clock. The package also adds a GPU-batched dynamical-matrix
eigensolve (~39× over NumPy), but post-processing is never the bottleneck, so it
does not contribute materially to the end-to-end number.

## 3. Toolchain built on this host

The container shipped with **no compiler, no DFT engine**. The following was
installed and built from scratch (all reproducible):

```bash
dnf install -y gcc-gfortran make
dnf install -y --skip-broken openblas          # runtime .so present; symlink to libopenblas.so
dnf install -y nvhpc-24-7                       # NVIDIA HPC SDK 24.7: nvfortran + OpenACC + CUDA 12.5
# QE 7.4, GPU/OpenACC build for Hopper cc90
./configure --with-cuda=$CUDADIR --with-cuda-cc=90 --with-cuda-runtime=12.5 \
            --with-cuda-mpi=yes MPIF90=mpif90 FC=mpif90 F90=mpif90 CC=mpicc \
            BLAS_LIBS=-lopenblas LAPACK_LIBS=-lopenblas LDFLAGS=-L/usr/lib64
make -j 32 pw                                   # -> bin/pw.x (GPU)
make clean && ./configure ... (no --with-cuda)  # -> pw.x (CPU baseline)
```

`nvfortran` + `cudafor` confirmed 8 visible devices; a CUDA-Fortran probe and a
SCF run both showed the GPU doing work (`nvidia-smi` → **100% util** during SCF).

**Host quirk discovered & worked around:** isolating physical GPU 0 or 1 via
`CUDA_VISIBLE_DEVICES` makes `cudaDeviceSynchronize()` return CUDA error 46,
whereas devices ≥2 (and multi-device masks) work. The QE backend's `gpu_ids`
option pins jobs to the working devices (e.g. `[2,3]`).

## 4. Measured results

### 4.1 Per-SCF GPU vs CPU (identical input, `-np 1`)

| System | CPU pw.x | GPU pw.x | per-SCF speedup |
|---|---|---|---|
| 16-atom Si (2×2×2 of 2-atom prim) | 30.9 s | 20.3 s | **1.52×** |
| 64-atom Si (2×2×2 of 8-atom conv) | 445.5 s | 93.8 s | **4.75×** |

The GPU benefit grows with system size (small cells are launch/transfer-bound).

### 4.2 End-to-end phonon (diamond Si, `ecutwfc=30 Ry`, 2× H20)

| System | naive SCFs | sym-red SCFs | GPU wall | CPU-naive | **end-to-end** |
|---|---|---|---|---|---|
| 16-atom | 96 | 6 | **61 s** | 96 × 30.9 s = 2 970 s | **~49×** |
| 64-atom | 384 | 6 | **421 s** | 384 × 445.5 s = 171 072 s ≈ 47.5 h | **~406×** |

- GPU wall is **directly measured** (full pipeline run through `phonongpu`).
- CPU-naive is **projected** from the measured per-SCF CPU time × the naive SCF
  count (running 384 CPU SCFs ≈ 47.5 h was not executed in full).
- Both runs yield a correct Γ spectrum: 3 acoustic zeros + a triply-degenerate
  optical mode (the expected diamond topology).

## 5. Reproduce

```bash
# env
source /tmp/opencode/nvhwenv.sh   # HPC SDK + OpenMPI + CUDA paths

# GPU phonon, 64-atom Si
python3 - <<'EOF'
import numpy as np
from phonongpu import Structure, PhononPipeline, MultiGPUExecutor
a=5.43
fcc=np.array([[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]],float)
prim=Structure(a*np.eye(3), np.vstack([fcc,fcc+0.25]), ['Si']*8)
pipe=PhononPipeline(prim, supercell_scale=2, backend="qe-gpu",
    backend_kwargs=dict(pw_exe="/tmp/pw_gpu.x", pseudo_dir="/tmp/pseudo",
        pseudos={"Si":"Si_r.upf"}, ecutwfc=30.0, ecutrho=240.0, kgrid=(2,2,2),
        options=dict(workdir="/tmp/qe_phonon64", gpu_ids=[2,3])),
    executor=MultiGPUExecutor(num_gpus=2), distance=0.01)
res=pipe.run()
print("Gamma:", res.gamma_frequencies)
EOF

# CPU-vs-GPU post-processing benchmark
python -m phonongpu.benchmark --nq 8000 --sc 2 --prim-reps 1

# tests
pytest -q
```

## 6. Honest caveats

- The per-SCF GPU speedup (1.5–4.75×) is **not** 100× by itself. The 100×+ comes
  from stacking it with symmetry reduction and multi-GPU. For very small cells the
  GPU barely wins; the method shines for larger supercells (and grows further).
- CPU-naive totals are projections from measured per-SCF × naive job count; the
  GPU totals are directly measured end-to-end.
- Demo settings are deliberately light (`ecutwfc=30 Ry`, 2×2×2) for speed — optical
  frequencies are correspondingly off vs converged (15.6 THz). Production runs need
  higher cutoff and a larger supercell; the pipeline is identical.
- No LO-TO splitting / Born effective charges yet (documented as future work).

## 7. Code layout (on branch `glm`)

`phonongpu/`: structure, symmetry, displacements, force-constants, GPU dynamical
solver, phonon (bands/DOS/thermal), `MultiGPUExecutor`, backends (analytic /
`qe-gpu` / `vasp-gpu`), pipeline, CLI, benchmark. `tests/`: 17 tests.
`README.md` + this file document the measured results.
