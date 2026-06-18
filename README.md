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

> **Scope note on "100×".** End-to-end 100× is *only* reachable once a GPU build of
> a DFT engine (e.g. QE `pw.x` GPU, VASP GPU) is installed, because the SCF force
> evaluations — not the post-processing — dominate phonon DFT cost. This package
> provides the **concurrency (multi-GPU), the symmetry reduction, and the GPU
> post-processing**, and wires up real input writers/launchers for QE-GPU and
> VASP-GPU. The host this was developed on has **no DFT engine, no Fortran/nvcc, no
> MPI**, so the per-SCF GPU speedup is *not demonstrated here*; the number below is
> the genuinely measured post-processing speedup.

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
| SCF force evaluations | per-SCF GPU speedup | **needs GPU DFT engine installed** | not on this host |
| SCF force evaluations | multi-GPU concurrency | implemented (QE/VASP backends) | scheduler tested |
| FC → dynamical matrix | batched build + Hermitian eigensolve on GPU | implemented + tested | **~39× measured** |

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
