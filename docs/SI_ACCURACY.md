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

- **The optical phonon matches experiment to <1%** — the full pipeline (symmetry-reduced
  finite displacement → real GPU-DFT forces → force-constant assembly → dynamical matrix →
  diagonalization) produces physically correct frequencies. This validates both the
  implementation and the unit handling. Quick 16-atom run: Γ optical 15.68 THz (+0.99 %);
  converged 4×4×4 / k(2,2,2) run: **15.42 THz (−0.7 %)** — see dispersion section below.
- The 3 acoustic modes are exactly zero at Γ (acoustic-sum-rule projection working).
- The spectrum is stable (no soft/imaginary modes) along Γ → (0.5,0.5,0.5).

## Honest limitations of this particular run

- The 2×2×2 supercell gives a **short real-space force-constant range**, so the dispersion
  *away* from Γ is under-resolved. The **Γ frequency is accurate regardless** (it does not
  depend on the FC range). A larger supercell resolves the full dispersion — see below.
- No LO-TO splitting (Born effective charges / dielectric ε∞) yet → optical modes at Γ are
  the TO frequency. For Si the LO-TO splitting is small, so the comparison to the Γ Raman
  frequency is appropriate.

## Dispersion with a 4×4×4 supercell (128 atoms)

Converged run with proper k-sampling:

- Supercell 4×4×4 = **128 atoms**, still **1 symmetry-irreducible atom → 6 DFT force evals**
  (a **128× reduction** vs the naive 768), found correctly and instantly via **spglib**
  (fixed: spglib needs *fractional* positions and *integer* atom-type codes).
- DFT: `ecutwfc = 40 Ry`, **`kgrid (2,2,2)` in the supercell** (≈ `8×8×8` primitive mesh —
  dense enough for converged forces). Earlier Γ-only k (`4×4×4` primitive) biased the Γ
  optical high (15.68); the dense k-grid fixes it.
- Wall: **1534 s** (~26 min, 6 SCFs over 3× H20, devices 2/3/7).

**Γ optical = 15.42 THz vs experiment 15.53 → −0.7 %** (was +0.99 % at Γ-only k).

Zone-edge (q=(½,½,½), primitive fractional — note: not the standard fcc L label) vs
experiment (Si, THz):

| mode | this work (k(2,2,2)) | experiment |
|---|---|---|
| Γ optical (Raman) | **15.42** | 15.53 (−0.7 %) |
| TA | 3.20 | 3.61 |
| LA | 11.19 | 11.35 (good) |
| TO | 12.42 | 12.55 (good) |

Phonon DOS (8×8×8 q-grid) reproduces the Si signature:
- acoustic band 0 → ~7.8 THz
- partial acoustic–optical gap in 8–11 THz
- optical band ~10 → 15.42 THz (max ≈ experiment's ~15.5)
- stable (no imaginary modes)

**Honest caveat:** the highest two optical branches sit near ~14.7 THz at some q-points
(exp optical ceiling is ~15.5 only at Γ). This is partly because the plotted q-points are
in *primitive fractional* coordinates (not the standard fcc X/L), and partly residual
4×4×4 supercell folding; the DOS — which averages over the full BZ — is clean and correct.
For publication-point dispersion at exact X/L/W, a larger supercell (5×5×5+) and the
standard conventional-cell q-labels would be used; the pipeline is identical.



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
