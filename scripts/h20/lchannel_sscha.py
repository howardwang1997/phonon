"""(L)-channel TMD-family screen — the Paper-2 discovery seed.

Generalises the proven NbSe2 SSCHA recipe (scripts/vq3e_nbse2_sscha.py) to an
arbitrary 2H/1T transition-metal-dichalcogenide monolayer, driven by an MLIP
(foundation or FC-distilled). Two modes:

  --mode triage : cheap harmonic finite-displacement dispersion (G-M-K-G).
                  Records the global min frequency + the q where it occurs.
                  Run for EVERY family member (incl. gapped controls) to RANK
                  which monolayers show any (L)-channel softening tendency.

  --mode sscha  : full SSCHA free-energy-Hessian T-evolution. The auxiliary
                  harmonic dyn is positive-definite by construction; the
                  free-energy Hessian gives the physical T-dependent phonons
                  that can soften -> 0 at an (L)-driven CDW. Run only for the
                  known/likely-CDW subset (config: sscha=true).

The screen is MLIP-only (no DFT) so it is a TRIAGE, not the final word:
foundation MLIPs are known to MISS electronically-driven soft modes (e.g.
foundation MACE finds NbSe2 dynamically stable). That is exactly the point of
the (E)/(L) decomposition — a material that experiment shows is CDW but that
stays *(L)-stable* here is an electronic-driven candidate, i.e. worth the FP64
(E)-channel DFPT/EPW rental. See docs/H20_EXPERIMENT_PLAN.md §3.

    python scripts/h20/lchannel_sscha.py --mode triage --name NbSe2 \
        --formula NbSe2 --polytype 2H --a 3.44 --thickness 3.34 \
        --model medium --supercell 4,4,1 --out results/h20/lchannel/NbSe2_triage_medium.csv
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np   # noqa: E402

RY_TO_CM = 109736.75
EV_A2_TO_RY_BOHR2 = (1.0 / 13.605693009) / (1.8897259886 ** 2)


# --------------------------------------------------------------------------- #
# general 2H/1T TMD monolayer builder + clean relax
# --------------------------------------------------------------------------- #
def build_tmd(formula: str, polytype: str, a: float, thickness: float,
              vacuum: float = 7.5):
    from ase.build import mx2
    at = mx2(formula=formula, kind=polytype, a=a, thickness=thickness, vacuum=vacuum)
    at.pbc = True
    return at


def relax_and_clean(atoms, calc, formula, polytype, vacuum=7.5):
    """In-plane relax (hold vacuum) then rebuild a perfectly symmetric primitive
    at the relaxed (a, thickness) — the relaxer's in-plane shear otherwise trips
    phonopy/CC symmetry enumeration (see td_common / SSCHA gotchas)."""
    import td_common as tdc
    rel, info = tdc.relax_monolayer(atoms, calc, fmax=1e-3, steps=200)
    a_rel = float(np.linalg.norm(rel.cell[0]))
    z = rel.get_positions()[:, 2]
    th_rel = float(z.max() - z.min())
    clean = build_tmd(formula, polytype, a_rel, max(th_rel, 0.5), vacuum=vacuum)
    return clean, a_rel, th_rel, info


# --------------------------------------------------------------------------- #
# MLIP finite-displacement fc2 -> CellConstructor dyn  (reused from vq3e)
# --------------------------------------------------------------------------- #
def mlip_fc2_phonon(atoms, calc, supercell, displacement):
    from phonon_accel.phonons import PhononCalculation
    sc = np.diag(supercell) if np.asarray(supercell).ndim == 1 else np.asarray(supercell)
    phon = PhononCalculation(atoms, supercell_matrix=sc,
                             primitive_matrix=np.eye(3), displacement=displacement)
    phon.compute_forces(calc)
    phon.produce_force_constants(symmetrize=True)
    return phon.phonon


def phonon_to_cc_dyn(ph):
    if not hasattr(np, "int"):
        np.int = int
        np.float = float
    import cellconstructor as CC
    import cellconstructor.ForceTensor   # noqa: F401
    import cellconstructor.Structure     # noqa: F401
    from ase import Atoms

    nat_sc = len(ph.supercell)
    fc = np.asarray(ph.force_constants)
    if fc.shape[0] != nat_sc or fc.shape[1] != nat_sc:
        raise RuntimeError(f"expected full fc ({nat_sc},{nat_sc},3,3), got {fc.shape}")
    M = fc.transpose(0, 2, 1, 3).reshape(3 * nat_sc, 3 * nat_sc) * EV_A2_TO_RY_BOHR2
    prim = ph.primitive
    uc = Atoms(numbers=prim.numbers, scaled_positions=prim.scaled_positions,
               cell=prim.cell, pbc=True)
    struc = CC.Structure.Structure()
    struc.generate_from_ase_atoms(uc)
    scm = np.array(ph.supercell_matrix)
    dim = np.array([scm[i, i] for i in range(3)], dtype=np.intc)
    sc = struc.generate_supercell(dim)
    t2 = CC.ForceTensor.Tensor2(struc, sc, dim)
    t2.SetupFromTensor(M)
    dyn = t2.GeneratePhonons(dim)
    return dyn


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def run_triage(a) -> int:
    import td_common as tdc
    calc = tdc.get_calc(a.model_type, a.model, device=a.device)
    atoms = build_tmd(a.formula, a.polytype, a.a, a.thickness)
    clean, a_rel, th_rel, info = relax_and_clean(atoms, calc, a.formula, a.polytype)
    sc = [int(x) for x in a.supercell.split(",")]
    disp = tdc.dispersion(clean, calc, supercell=sc, displacement=a.displacement,
                          path="GMKG", npoints=121)
    freq = disp["frequencies"]                 # (n_q, n_band) THz
    qf = disp["qpoints_frac"]
    flat = freq.min(axis=1)                    # lowest band per q
    iq = int(np.argmin(flat))
    minf = float(flat[iq])
    qmin = qf[iq]
    label = "soft" if minf < -0.1 else ("marginal" if minf < 0.3 else "stable")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    hdr = "name,polytype,model,a_relaxed,thickness,harm_minfreq_THz,qmin_a,qmin_b,qmin_c,label"
    out.write_text(
        hdr + "\n" +
        f"{a.name},{a.polytype},{a.tag},{a_rel:.4f},{th_rel:.3f},{minf:.4f},"
        f"{qmin[0]:.4f},{qmin[1]:.4f},{qmin[2]:.4f},{label}\n")
    print(f"[triage:{a.name}/{a.tag}] a={a_rel:.3f} min_harm_freq={minf:.3f} THz "
          f"@ q=({qmin[0]:.2f},{qmin[1]:.2f},{qmin[2]:.2f}) -> {label}", flush=True)
    return 0


def run_sscha(a) -> int:
    import td_common as tdc
    from sscha.Ensemble import Ensemble
    from sscha.SchaMinimizer import SSCHA_Minimizer
    from sscha.Relax import SSCHA

    calc = tdc.get_calc(a.model_type, a.model, device=a.device)
    atoms = build_tmd(a.formula, a.polytype, a.a, a.thickness)
    clean, a_rel, th_rel, info = relax_and_clean(atoms, calc, a.formula, a.polytype)
    sc = [int(x) for x in a.supercell.split(",")]
    ph = mlip_fc2_phonon(clean, calc, sc, a.displacement)
    dyn = phonon_to_cc_dyn(ph)
    w_bare, _ = dyn.DiagonalizeSupercell()
    bare = float(w_bare.min() * RY_TO_CM)
    print(f"[sscha:{a.name}/{a.tag}] a={a_rel:.3f}  bare-MLIP-fc2 min freq = "
          f"{bare:.1f} cm^-1 ({'SOFT' if bare < -1 else 'stable'})", flush=True)
    dyn.ForcePositiveDefinite()
    dyn.Symmetrize()
    supercell = dyn.GetSupercell()
    temps = [float(x) for x in a.temperatures.split(",")]

    rows = ["T_K,sscha_minfreq_cm,n_imag"]
    tcross = None
    minf_hi = None
    for T in temps:
        t0 = time.perf_counter()
        ens = Ensemble(dyn, T0=T, supercell=supercell)
        minim = SSCHA_Minimizer(ens)
        minim.min_step_dyn = 0.05
        minim.kong_liu_ratio = 0.5
        relax = SSCHA(minim, ase_calculator=calc, N_configs=a.nconfigs, max_pop=a.maxpop)
        relax.relax(get_stress=False)
        hess = relax.minim.ensemble.get_free_energy_hessian()
        wh, _ = hess.DiagonalizeSupercell()
        wcm = wh * RY_TO_CM
        minf = float(wcm.min())
        n_imag = int((wcm < -1.0).sum())
        rows.append(f"{T:.0f},{minf:.2f},{n_imag}")
        if minf > 0 and tcross is None and bare < -1:
            tcross = T
        minf_hi = minf
        print(f"  T={T:.0f} K: SSCHA min freq = {minf:.1f} cm^-1 (n_imag={n_imag}); "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
        dyn = relax.minim.dyn   # warm-start next T

    # classification for the origin map
    if bare >= -1 and (minf_hi is None or minf_hi >= -1):
        label = "L-stable"        # MLIP+SSCHA sees no lattice instability
    elif minf_hi is not None and minf_hi < -1:
        label = "L-unstable"      # stays soft up to the top T -> strong (L)
    elif tcross is not None:
        label = f"L-crossover@{int(tcross)}K"
    else:
        label = "L-marginal"

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = (f"# name={a.name} polytype={a.polytype} model={a.tag} a_relaxed={a_rel:.4f} "
            f"bare_minfreq_cm={bare:.2f} cdw_exp_K={a.cdw_exp_K} label={label}\n")
    out.write_text(meta + "\n".join(rows) + "\n")
    print(f"[sscha:{a.name}/{a.tag}] label={label}  (exp T_CDW={a.cdw_exp_K} K)  "
          f"-> {out.relative_to(ROOT)}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["triage", "sscha"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--formula", required=True)
    ap.add_argument("--polytype", default="2H", choices=["2H", "1T"])
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--thickness", type=float, required=True)
    ap.add_argument("--model-type", dest="model_type", default="mace")
    ap.add_argument("--model", required=True, help="foundation tag (small/medium) or a .model path")
    ap.add_argument("--tag", default=None, help="label for the model column (default = basename of --model)")
    ap.add_argument("--supercell", default="4,4,1")
    ap.add_argument("--displacement", type=float, default=0.03)
    ap.add_argument("--temperatures", default="50,100,150,200,300")
    ap.add_argument("--nconfigs", type=int, default=200)
    ap.add_argument("--maxpop", type=int, default=5)
    ap.add_argument("--cdw-exp-K", dest="cdw_exp_K", default="null")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.tag is None:
        a.tag = Path(a.model).stem if "/" in a.model or a.model.endswith(".model") else a.model
    return run_triage(a) if a.mode == "triage" else run_sscha(a)


if __name__ == "__main__":
    raise SystemExit(main())
