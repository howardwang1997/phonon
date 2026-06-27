"""#1 (rigorous C): NbSe2 CDW soft-mode T-evolution via SSCHA (free-energy
Hessian) with the distilled-FT MLIP. SSCHA is the rigorous tool near an
instability (TDEP's perturbative footing is weak there): the auxiliary harmonic
dyn is positive-definite by construction, and the *free-energy Hessian* gives
the physical T-dependent phonons that can soften -> 0 at the CDW transition.

Starting dyn = the NbSe2 DFT fc2 (a=3.44, 3x3) made positive-definite; ensembles
are evaluated with the FT MLIP (same model as the TDEP run C).

    LD_LIBRARY_PATH=$CONDA/envs/phonon/lib python scripts/vq3e_nbse2_sscha.py \
        --phonopy results/vq3/nbse2_dft_phonopy.yaml \
        --model results/finetune_nbse2/ft_nbse2.model \
        --temperatures 20,100,200,300 --nconfigs 300 --maxpop 5 --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import td_common as tdc

RY_TO_CM = 109736.75


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phonopy", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperatures", default="20,100,200,300")
    ap.add_argument("--nconfigs", type=int, default=300)
    ap.add_argument("--maxpop", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="nbse2_sscha")
    ap.add_argument("--outdir", default="results/td_phonon")
    a = ap.parse_args()

    if not hasattr(np, "int"):
        np.int = int
        np.float = float
    import cellconstructor as CC
    import cellconstructor.Phonons
    import cellconstructor.ForceTensor
    import cellconstructor.Structure
    import phonopy
    from ase import Atoms
    from sscha.Ensemble import Ensemble
    from sscha.SchaMinimizer import SSCHA_Minimizer
    from sscha.Relax import SSCHA

    temps = [float(x) for x in a.temperatures.split(",")]

    # --- build the starting CC dyn from the DFT fc2 (full supercell FC -> Tensor2) ---
    # CC wants the diagonal supercell size [nx,ny,nz] (NOT the 3x3 matrix), and the
    # full (3*nat_sc, 3*nat_sc) FC in Ry/Bohr^2.
    EV_A2_TO_RY_BOHR2 = (1.0 / 13.605693009) / (1.8897259886 ** 2)
    ypath = str(ROOT / a.phonopy) if not Path(a.phonopy).is_absolute() else a.phonopy
    ph = phonopy.load(ypath, is_compact_fc=False)
    nat_sc = len(ph.supercell)
    fc = np.asarray(ph.force_constants)                    # (nat_sc, nat_sc, 3, 3) eV/A^2
    M = fc.transpose(0, 2, 1, 3).reshape(3 * nat_sc, 3 * nat_sc) * EV_A2_TO_RY_BOHR2
    prim = ph.primitive
    uc = Atoms(numbers=prim.numbers, scaled_positions=prim.scaled_positions,
               cell=prim.cell, pbc=True)
    struc = CC.Structure.Structure(); struc.generate_from_ase_atoms(uc)
    scm = np.array(ph.supercell_matrix)
    dim = np.array([scm[i, i] for i in range(3)], dtype=np.intc)
    sc = struc.generate_supercell(dim)
    t2 = CC.ForceTensor.Tensor2(struc, sc, dim)
    t2.SetupFromTensor(M)
    dyn = t2.GeneratePhonons(dim)
    w_bare, _ = dyn.DiagonalizeSupercell()
    print(f"[sscha] bare DFT-fc2 dyn: supercell {dyn.GetSupercell()}, min freq "
          f"{w_bare.min()*RY_TO_CM:.1f} cm^-1 (soft mode if < 0)", flush=True)
    dyn.ForcePositiveDefinite()    # SSCHA needs a positive-definite trial dyn
    dyn.Symmetrize()
    w0, _ = dyn.DiagonalizeSupercell()
    print(f"[sscha] starting (pos-def) min freq = {w0.min()*RY_TO_CM:.1f} cm^-1", flush=True)

    calc = tdc.get_calc("mace", a.model, device=a.device)
    supercell = dyn.GetSupercell()

    rows = ["T_K,sscha_minfreq_cm,n_imag,n_pop"]
    out = {"temperatures": np.array(temps), "tag": np.array(a.tag)}
    for T in temps:
        t0 = time.perf_counter()
        print(f"[sscha] === T={T:.0f} K: SSCHA relax (N={a.nconfigs}, max_pop={a.maxpop}) ===",
              flush=True)
        ens = Ensemble(dyn, T0=T, supercell=supercell)
        minim = SSCHA_Minimizer(ens)
        minim.min_step_dyn = 0.05
        minim.kong_liu_ratio = 0.5
        relax = SSCHA(minim, ase_calculator=calc, N_configs=a.nconfigs, max_pop=a.maxpop)
        relax.relax(get_stress=False)
        # free-energy Hessian = the physical T-dependent phonons
        hess = relax.minim.ensemble.get_free_energy_hessian()
        wh, _ = hess.DiagonalizeSupercell()
        wcm = wh * RY_TO_CM
        n_imag = int((wcm < -1.0).sum())
        minf = float(wcm.min())
        npop = getattr(relax, "pop", getattr(relax, "__pop", a.maxpop))
        rows.append(f"{T:.0f},{minf:.2f},{n_imag},{npop}")
        out[f"T{int(T)}_freqs"] = wcm
        print(f"[sscha] T={T:.0f} K: SSCHA free-energy-Hessian min freq = {minf:.1f} cm^-1 "
              f"(n_imag={n_imag}); {time.perf_counter()-t0:.0f}s", flush=True)
        # warm-start next T from this dyn
        dyn = relax.minim.dyn

    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.tag}.csv").write_text("\n".join(rows) + "\n")
    np.savez(outdir / f"{a.tag}.npz", **out)
    print("[sscha] wrote", (outdir / f"{a.tag}.csv").relative_to(ROOT), flush=True)
    print("\n".join(rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
