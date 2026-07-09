"""SSCHA free-energy Hessian -> phonon BAND STRUCTURE along Gamma-M-K-Gamma at each T_lat.
Same as vq3e_nbse2_sscha.py but, at each T, interpolates the free-energy Hessian (CC Phonons
object) onto a q-path via DiagonalizeQ and saves the full dispersion (not just the supercell
min). This gives the (L)-temperature phonon SPECTRA.

Run on 2060:
  python scripts/smearing_kink/vq3e_sscha_bands.py \
    --phonopy <fc2 yaml> --model <path-p ft.model> --tag <mat>_Lband --device cuda
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import td_common as tdc
RY_TO_CM = 109736.75
EV_A2_TO_RY_BOHR2 = (1.0 / 13.605693009) / (1.8897259886 ** 2)


def tmd_qpath(nseg=25):
    """Gamma-M-K-Gamma in fractional (2pi/a), 2D. Returns qs (nq,3), seg_x, ticks, labels."""
    pts = [(0, 0, 0), (0.5, 0, 0), (1.0 / 3.0, 1.0 / 3.0, 0), (0, 0, 0)]
    labels = ["$\\Gamma$", "M", "K", "$\\Gamma$"]
    qs, x, ticks = [], [0.0], [0.0]
    for i in range(len(pts) - 1):
        q0 = np.array(pts[i]); q1 = np.array(pts[i + 1]); seg = np.linalg.norm(q1 - q0)
        for j in range(1, nseg + 1):
            qs.append(q0 + (q1 - q0) * j / nseg)
            x.append(x[-1] + seg / nseg)
        ticks.append(x[-1])
    return np.array(qs), np.array(x), ticks, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phonopy", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperatures", default="20,100,200,300")
    ap.add_argument("--nconfigs", type=int, default=300)
    ap.add_argument("--maxpop", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="sscha_bands")
    ap.add_argument("--outdir", default="results/td_phonon")
    a = ap.parse_args()
    if not hasattr(np, "int"):
        np.int = int; np.float = float
    import cellconstructor as CC
    import cellconstructor.Phonons, cellconstructor.ForceTensor, cellconstructor.Structure
    import phonopy
    from ase import Atoms
    from sscha.Ensemble import Ensemble
    from sscha.SchaMinimizer import SSCHA_Minimizer
    from sscha.Relax import SSCHA

    temps = [float(x) for x in a.temperatures.split(",")]
    ypath = str(ROOT / a.phonopy) if not Path(a.phonopy).is_absolute() else a.phonopy
    ph = phonopy.load(ypath, is_compact_fc=False)
    nat_sc = len(ph.supercell)
    fc = np.asarray(ph.force_constants)
    M = fc.transpose(0, 2, 1, 3).reshape(3 * nat_sc, 3 * nat_sc) * EV_A2_TO_RY_BOHR2
    prim = ph.primitive
    uc = Atoms(numbers=prim.numbers, scaled_positions=prim.scaled_positions, cell=prim.cell, pbc=True)
    struc = CC.Structure.Structure(); struc.generate_from_ase_atoms(uc)
    scm = np.array(ph.supercell_matrix); dim = np.array([scm[i, i] for i in range(3)], dtype=np.intc)
    sc = struc.generate_supercell(dim)
    t2 = CC.ForceTensor.Tensor2(struc, sc, dim); t2.SetupFromTensor(M)
    dyn = t2.GeneratePhonons(dim); dyn.ForcePositiveDefinite(); dyn.Symmetrize()
    calc = tdc.get_calc("mace", a.model, device=a.device)
    supercell = dyn.GetSupercell()

    qs, xticks_x, ticks, labels = tmd_qpath(25)
    rows = ["T_K,minfreq_cm,n_imag"]
    out = {"qs": qs, "x": xticks_x, "ticks": ticks, "labels": labels, "tag": a.tag, "temperatures": np.array(temps)}
    for T in temps:
        t0 = time.perf_counter()
        print(f"[sscha-band] === T={T:.0f} K: SSCHA ===", flush=True)
        dyn.ForcePositiveDefinite(); dyn.Symmetrize()
        ens = Ensemble(dyn, T0=T, supercell=supercell)
        minim = SSCHA_Minimizer(ens); minim.min_step_dyn = 0.05; minim.kong_liu_ratio = 0.5
        relax = SSCHA(minim, ase_calculator=calc, N_configs=a.nconfigs, max_pop=a.maxpop)
        relax.relax(get_stress=False)
        hess = relax.minim.ensemble.get_free_energy_hessian()
        wh, _ = hess.DiagonalizeSupercell(); wcm = wh * RY_TO_CM
        minf = float(wcm.min()); n_imag = int((wcm < -1.0).sum())
        # band structure: interpolate free-energy Hessian onto Gamma-M-K-Gamma
        bands = []
        for q in qs:
            try:
                wq, _ = hess.DiagonalizeQ(q); bands.append(wq * RY_TO_CM)
            except Exception:
                bands.append(np.full(len(wh), np.nan))
        out[f"T{int(T)}_bands"] = np.array(bands)
        rows.append(f"{T:.0f},{minf:.2f},{n_imag}")
        print(f"[sscha-band] T={T:.0f} K: min={minf:.1f} cm^-1 n_imag={n_imag}; bands {len(qs)} q; {time.perf_counter()-t0:.0f}s", flush=True)
        dyn = relax.minim.dyn

    outdir = ROOT / a.outdir; outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.tag}.csv").write_text("\n".join(rows) + "\n")
    np.savez(outdir / f"{a.tag}.npz", **out)
    print("[sscha-band] wrote", outdir / f"{a.tag}.npz", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
