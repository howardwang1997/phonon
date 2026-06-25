# Building GPU Quantum ESPRESSO (NVHPC) — recipe

Built and verified on a rented **1× Tesla V100-SXM2-32GB** (Ubuntu 22.04, driver 550,
CUDA 12.4, gcc 11.4). Result: `pw.x` with `-cuda -gpu=cc70,cuda12.4`, **~19× faster
than the 8-core CPU build** (50-atom graphene SCF: ~32 min → ~100 s; 2-atom: 67 s → 3.8 s),
energies matching CPU to 7 digits. Makes Path P scaling (100s of DFT single-points) and
M3's NbSe₂ DFT distillation feasible on a single GPU.

## 1. Toolchain — NVHPC 24.5 (cuda 12.4)
Pick the NVHPC whose bundled CUDA **matches the driver** (driver 550 → CUDA ≤12.4; 12.6
won't run). NVHPC 24.5 bundles cuda 12.4 + nvfortran + an MPI built with nvfortran.

```bash
tar xpzf nvhpc_2024_245_Linux_x86_64_cuda_12.4.tar.gz
cd nvhpc_2024_245_Linux_x86_64_cuda_12.4
NVHPC_SILENT=true NVHPC_INSTALL_DIR=/opt/nvidia/hpc_sdk NVHPC_INSTALL_TYPE=single ./install
```

Environment (`/root/nvhpc_env.sh`) — source before configure, build, AND run:
```bash
export NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5
export PATH=$NVHPC/compilers/bin:$NVHPC/comm_libs/mpi/bin:$NVHPC/cuda/12.4/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC/compilers/lib:$NVHPC/cuda/12.4/lib64:$NVHPC/comm_libs/mpi/lib:$NVHPC/math_libs/lib64:$LD_LIBRARY_PATH
```

## 2. Configure QE 7.3.1
```bash
source /root/nvhpc_env.sh
cd q-e-qe-7.3.1
./configure --with-cuda=$NVHPC/cuda/12.4 --with-cuda-cc=70 --with-cuda-runtime=12.4 \
            --enable-openmp --with-scalapack=no
# expect DFLAGS ... -D__CUDA  (GPU on). cc=70 is V100; use 80 for A100, 90 for H100.
```

## 3. The submodule trap (this is what bites)
The GitHub **archive tarball** (`.../archive/refs/tags/qe-7.3.1.tar.gz`) ships **no
submodule contents and no submodule commit hashes** (only gipaw+wannier90 are recorded).
The build then tries `git fetch origin <SHA>` for devxlib/fox, which **gitlab/github
refuse by default** (`allowReachableSHA1InWant` off) → `FETCH_HEAD is not a commit`.

Fix — full-clone the libs, then pin to QE 7.3.1's **exact** commit (a full clone has all
history; the macro skips its broken fetch once `external/<sub>/.git` exists):
```bash
cd q-e-qe-7.3.1
rm -rf external/devxlib external/fox
git clone https://gitlab.com/max-centre/components/devicexlib.git external/devxlib
git clone https://github.com/pietrodelugas/fox.git            external/fox
for sub in devxlib fox; do
  SHA=$(wget -qO- "https://api.github.com/repos/QEF/q-e/contents/external/$sub?ref=qe-7.3.1" \
        | grep -oE '"sha": "[a-f0-9]{40}"' | head -1 | grep -oE '[a-f0-9]{40}')
  (cd external/$sub && git checkout -q "$SHA")
done
rm -f install/libcuda_devxlib install/libfox   # clear any stale build stamps
```
**Why it matters:** the *latest* devicexlib dropped the `device_fbuff_m` module that
LAXlib's `cdiaghg.f90`/`rdiaghg.f90` (the GPU generalized eigensolver) still `use`s →
`NVFORTRAN-F-0004-Unable to open MODULE file device_fbuff_m.mod`. The pinned SHA
(`a6b89ef…` for QE 7.3.1) still has `device_fbuff_mod.f90`.

## 4. Build + run
```bash
source /root/nvhpc_env.sh
make -j8 pw                                   # ~30–40 min; links bin/pw.x
source /root/nvhpc_env.sh
mpirun --allow-run-as-root -np 1 /root/q-e-qe-7.3.1/bin/pw.x -in scf.in
# 1 MPI rank per GPU. Output should say "GPU acceleration is ACTIVE."
```

## 5. Gotcha for graphene inputs
Graphene is a semimetal → a bare SCF can die with `c_bands (1): … stopping`. Give it
enough empty bands and a non-coarse k-mesh (e.g. `nbnd≈3×atoms`, `mixing_beta 0.3`,
6×6 k for a 5×5 cell). ASE's `PhononCalculation`/`Espresso` already set sane values;
hand-written inputs need this explicitly.
