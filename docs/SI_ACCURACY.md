# Si phonon accuracy validation

Real GPU-DFT phonon run via `phonongpu` driving a GPU build of Quantum ESPRESSO 7.4.
This is a physics-correctness check: do the frequencies agree with experiment?

## Setup

- System: diamond Si, 2-atom primitive (rhombohedral fcc).
- Supercell: 2×2×2 (16 atoms) → 1 symmetry-irreducible atom → **6 DFT force evals**.
- DFT: QE-GPU (`pw.x`, nvfortran/OpenACC, Hopper cc90), `Si_r.upf` (ONCV norm-conserving),
  `ecutwfc = 40 Ry`, `ecutrho = 160 Ry`, k-grid `(2,2,2)` in the supercell (≈ `4×4×4` primitive).
- Hardware: 3× H20 (devices 2, 3, 7 — the only isolated devices that work on this host).
- Displacement: 0.02 Å, central differences.
- Units: forces converted Ry/Bohr → eV/Å (×25.711); frequencies via the standard
  √(eV/Å²/amu) → ×15.633 → THz convention.

## Result

```
[pipeline] primitive atoms=2  supercell atoms=16
[pipeline] force evaluations=6  (naive full = 96; 16.0x fewer via symmetry)
wall time: 321 s (6 SCFs over 3 GPUs)

Gamma acoustic branches (THz) :  ~0.000  ~0.000  ~0.000      (3 translational zeros)
Gamma optical branch   (THz) :  15.684  15.684  15.684       (triply degenerate)
Experiment (Γ Raman)   (THz) :  15.53
Error                          :  +0.99 %
Stability (G→L path)           :  no imaginary modes (min ≈ 0)
```

## Interpretation

- **The optical phonon is within ~1% of experiment** — the full pipeline (symmetry-reduced
  finite displacement → real GPU-DFT forces → force-constant assembly → dynamical matrix →
  diagonalization) produces physically correct frequencies. This validates both the
  implementation and the unit handling.
- The 3 acoustic modes are exactly zero at Γ (acoustic-sum-rule projection working).
- The spectrum is stable (no soft/imaginary modes) along Γ → (0.5,0.5,0.5).

## Honest limitations of this particular run

- The 2×2×2 supercell gives a **short real-space force-constant range**, so the dispersion
  *away* from Γ is under-resolved (the G→L branch is nearly flat). The **Γ frequency is
  accurate regardless** (it does not depend on the FC range), but resolving the acoustic
  TA/LA and optical dispersion at X/L needs a larger supercell (4×4×4 or 5×5×5, i.e.
  128–250 atoms). Those larger SCFs are correct but cost ~10 min/SCF on this host with the
  current settings; they were not completed within the time budget here.
- No LO-TO splitting (Born effective charges / dielectric ε∞) yet → optical modes at Γ are
  the TO frequency. For Si the LO-TO splitting is small, so the comparison to the Γ Raman
  frequency is appropriate.
- Convergence: `ecutwfc=40 Ry` is near-converged for this pseudo; `k(2,2,2)` is adequate
  for Γ-force accuracy in this small cell.

## Reproduce

```bash
source /tmp/opencode/nvhwenv.sh   # HPC SDK + OpenMPI + CUDA paths
python3 - <<'EOF'
import numpy as np
from phonongpu import Structure, PhononPipeline, MultiGPUExecutor
from phonongpu.phonon import freqs_to_thz
a=5.43
lat=a/2*np.array([[0,1,1],[1,0,1],[1,1,0]],float)
prim=Structure(lat, [[0,0,0],[0.25,0.25,0.25]], ['Si','Si'])
pipe=PhononPipeline(prim, supercell_scale=2, backend="qe-gpu",
    backend_kwargs=dict(pw_exe="/tmp/pw_gpu.x", pseudo_dir="/tmp/pseudo",
        pseudos={"Si":"Si_r.upf"}, ecutwfc=40.0, ecutrho=160.0, kgrid=(2,2,2),
        options=dict(workdir="/tmp/qe_si_acc", gpu_ids=[2,3,7])),
    executor=MultiGPUExecutor(num_gpus=3), distance=0.02)
res=pipe.run()
print("Gamma optical (THz):", freqs_to_thz(np.sort(res.gamma_frequencies))[3:])
EOF
```
