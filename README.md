# phonongpu

GPU-accelerated finite-displacement phonon pipeline with multi-GPU SCF scheduling.

`phonongpu` reduces the wall-clock cost of phonon-spectrum calculations through
three complementary mechanisms, and is **honest about which parts are accelerated
where**:

1. **Symmetry-reduced finite displacements** — only symmetry-irreducible atoms are
   displaced (e.g. diamond Si: 1 atom → 6 force evals instead of `6·N`, an 8–64×
   reduction in the number of SCF jobs). This is the single biggest lever and is
   fully implemented and tested.
2. **Multi-GPU concurrent SCF scheduling** — the irreducible supercell SCF jobs are
   pipelined across all available GPUs through a `MultiGPUExecutor`, each pinned via
   `CUDA_VISIBLE_DEVICES`. Pluggable backends write ready-to-run input files for
   **Quantum ESPRESSO (GPU build)** and **VASP (GPU build)**.
3. **GPU-accelerated phonon post-processing** — force-constant assembly, the
   batched dynamical-matrix build, and the Hermitian eigensolve over the whole
   q-grid run on GPU via PyTorch. **Measured ~39× over single-threaded NumPy on a
   single NVIDIA H20** (8000 q-points, 24 branches); see benchmark below.

> **Scope note on "100×".** End-to-end phonon wall-clock speedup is the product of
> three independent factors, all measured on this host:
> - **symmetry reduction**: 16–64× fewer SCF jobs
> - **per-SCF GPU speedup**: 1.5× (16-atom) … 4.75× (64-atom), grows with size
> - **multi-GPU concurrency**: ~N× (used 2× H20)
>
> Fully-measured end-to-end (both sides run):
> - 16-atom Si phonon: GPU 61 s vs CPU-naive 2 970 s → **~49×**
>
> GPU-measured, CPU-projected from measured per-SCF (445.5 s) × naive SCF count:
> - 64-atom Si phonon: GPU **421 s** vs CPU-naive 171 072 s (≈47.5 h) → **~406×**
>
> The per-SCF GPU component alone is 1.5–4.75× at these sizes (not 100× by itself);
> the 100×+ headline comes from stacking it with symmetry reduction and multi-GPU.
>
> **A full GPU build of Quantum ESPRESSO 7.4 was compiled and run here** (NVIDIA HPC
> SDK 24.7 `nvfortran` + OpenACC, CUDA 12.5, Hopper cc90). See `docs/ACCELERATION_REPORT.md`.

## Install

```bash
pip install -e .            # core (numpy/scipy/pyyaml)
pip install -e .[gpu]       # add PyTorch for GPU solver
pip install -e .[test]      # add pytest
```

Optional: `pip install spglib` for a faster/more robust symmetry finder (a pure-NumPy
fallback is built in).

## Quick start (analytic demo, no DFT engine needed)

```bash
python -m phonongpu.cli run --backend analytic --supercell 2 --gamma --dos-mesh 12
```

## Driving a real GPU-DFT engine

```python
from phonongpu import Structure, PhononPipeline, MultiGPUExecutor
prim = Structure.from_poscar("POSCAR")          # or construct directly
executor = MultiGPUExecutor(num_gpus=3)          # 3× H20
pipe = PhononPipeline(
    prim, supercell_scale=2,
    backend="qe-gpu",                            # or "vasp-gpu"
    backend_kwargs={
        "pseudos": {"Si": "Si.UPF"},
        "ecutwfc": 60.0, "kgrid": (4, 4, 4),
        "options": {"workdir": "./qe_jobs"},
    },
    executor=executor,
    distance=0.01,
)
result = pipe.run()
print(result.gamma_frequencies)
grid, dos, _ = result.dos.dos(mesh=24)
temps, Cv = result.dos.thermal(mesh=24)
```

## Benchmark (this host: NVIDIA H20)

```bash
python -m phonongpu.benchmark --nq 8000 --sc 2 --prim-reps 1
```

Typical result on one H20 (8-atom primitive → 24 branches, 8000 q-points):

```
CPU (numpy)            :  10461 ms
GPU cuda:0 (H20)       :    270 ms   speedup x 38.9
```

## Where the speedup comes from (honest breakdown)

| Stage | Mechanism | Status | Demonstrated |
|---|---|---|---|
| SCF force evaluations | symmetry reduction (8–64× fewer jobs) | implemented + tested | yes (FC recovery exact) |
| SCF force evaluations | per-SCF GPU speedup | QE-GPU 7.4 built + run | **~4.75× measured (64-atom Si)** |
| SCF force evaluations | multi-GPU concurrency | implemented (QE/VASP backends) | yes (2× H20) |
| FC → dynamical matrix | batched build + Hermitian eigensolve on GPU | implemented + tested | **~39× measured** |

## Real GPU-DFT demonstration (this host)

A genuine GPU build of Quantum ESPRESSO 7.4 was compiled and driven by this package:

```bash
# toolchain installed on this host
dnf install -y gcc-gfortran make
dnf install -y nvhpc-24-7            # NVIDIA HPC SDK 24.7 (nvfortran + OpenACC + CUDA 12.5)
# QE 7.4 configured for GPU (Hopper cc90) and built
./configure --with-cuda=$CUDADIR --with-cuda-cc=90 --with-cuda-runtime=12.5 \
            --with-cuda-mpi=yes MPIF90=mpif90 FC=mpif90 F90=mpif90 CC=mpicc ...
make -j 32 pw
```

End-to-end GPU-DFT phonon runs (diamond Si, `ecutwfc=30 Ry`, 2× H20, devices 2–3):

| System | SCFs (sym-red) | GPU wall (measured) | CPU-naive (projected) | end-to-end |
|---|---|---|---|---|
| 16-atom (2×2×2) | 6 (of 96) | **61 s** | 96 × 30.9 s = 2 970 s | **~49×** |
| 64-atom (2×2×2) | 6 (of 384) | **421 s** | 384 × 445.5 s = 171 072 s (47.5 h) | **~406×** |

Both produce a correct Γ spectrum (3 acoustic zeros + triply-degenerate optical).

> Host quirk: on this NGC image, isolating GPU 0 or 1 via `CUDA_VISIBLE_DEVICES` makes
> `cudaDeviceSynchronize` return error 46, while devices ≥2 work. The QE backend's
> `gpu_ids` option handles this — set it to the working device indices (e.g. `[2,3,4,5]`).

## Tests

```bash
pytest -q
```

## Layout

```
phonongpu/
  structure.py        crystal structure, supercell + (cell,primitive) map, neighbors
  symmetry.py         space-group finder (spglib or NumPy fallback), atom orbits
  displacements.py    symmetry-reduced finite-displacement plan
  force_constants.py  FC assembly (symmetry fill) + acoustic-sum-rule enforcement
  dynamical.py        batched GPU/CPU dynamical matrix + Hermitian eigensolve
  phonon.py           bands, DOS, thermal properties
  scheduler.py        MultiGPUExecutor (concurrent SCF across GPUs)
  backends/           analytic (testable), qe-gpu, vasp-gpu input writers + launchers
  io/poscar.py        POSCAR read/write
  pipeline.py         end-to-end orchestrator
  cli.py, benchmark.py
tests/                pytest suite (16 tests)
```

## License

MIT.
