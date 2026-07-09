"""B2d Phase-2 — 2D kink(T_el, T_lat) surface via SSCHA through FriedelMACE.

ONE unified model = Path-P VSe2 backbone + T_el-conditioned Friedel LR term.
Sweep a (T_el, T_lat) grid; at each point run SSCHA (free-energy Hessian) THROUGH
FriedelMACECalculator -> soft-mode(T_el, T_lat). The (E) kink shows along the T_el
axis (Friedel amplitude decays), the (L) kink along the T_lat axis (backbone
anharmonic heal). This is the 2D realisation of the unified model.

Mirrors vq3e_nbse2_sscha.py but swaps the bare-MACE calc for FriedelMACECalculator
and nests a T_el loop around the T_lat SSCHA.

Run on 2060 (after M1 frees the GPU):
  conda run --no-capture-output -n phonon python scripts/smearing_kink/b2d_2d_sscha.py \
    --backbone results/vq_family/vse2/1T-VSe2_dg0.020.yaml \
    --template  results/vq_family/vse2/1T-VSe2_dg0.005.yaml \
    --model     results/finetune_path_p_1T-VSe2/ft.model --tag b2d_2d_vse2 --device cuda
"""
from __future__ import annotations
import argparse, csv, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import friedel_module as fm
from friedel_calc import FriedelCorrection, FriedelMACECalculator
from phonon_accel.phonons import phonopy_to_ase

SK = ROOT / "results" / "smearing_kink"
RY_TO_CM = 109736.75
EV_A2_TO_RY_BOHR2 = (1.0 / 13.605693009) / (1.8897259886 ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="VSe2 fc2 yaml dg0.020")
    ap.add_argument("--template", required=True, help="VSe2 fc2 yaml dg0.005")
    ap.add_argument("--model", required=True, help="Path-P VSe2 ft.model")
    ap.add_argument("--tel", default="789,2368,4737", help="T_el grid (K)")
    ap.add_argument("--tlat", default="20,110,200", help="T_lat grid (K)")
    ap.add_argument("--nconfigs", type=int, default=200)
    ap.add_argument("--maxpop", type=int, default=4)
    ap.add_argument("--tag", default="b2d_2d_vse2")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if not hasattr(np, "int"):
        np.int = int; np.float = float

    # --- B_law / kappa_law from friedel_family.csv (per-material VSe2 fit) ---
    rows = [r for r in csv.DictReader(open(SK / "friedel_family.csv")) if r["material"] == "1T-VSe2"]
    Tb = np.array([float(r["T_el"]) for r in rows]); Bb = np.array([float(r["B"]) for r in rows])
    Kb = np.array([float(r["kappa"]) for r in rows]); o = np.argsort(Tb)
    Tb, Bb, Kb = Tb[o], Bb[o], Kb[o]
    Blaw = lambda T: float(np.interp(T, Tb, Bb)); Klaw = lambda T: float(np.interp(T, Tb, Kb))

    # --- starting CC dyn from the backbone fc2 (positive-definite) ---
    import cellconstructor as CC
    import cellconstructor.Phonons, cellconstructor.ForceTensor, cellconstructor.Structure
    import phonopy
    from ase import Atoms
    from sscha.Ensemble import Ensemble
    from sscha.SchaMinimizer import SSCHA_Minimizer
    from sscha.Relax import SSCHA
    ypath = str(ROOT / a.backbone) if not Path(a.backbone).is_absolute() else a.backbone
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
    dyn0 = t2.GeneratePhonons(dim)
    w_bare, _ = dyn0.DiagonalizeSupercell()
    print(f"[b2d-2d] backbone bare min freq {w_bare.min()*RY_TO_CM:.1f} cm^-1", flush=True)
    dyn0.ForcePositiveDefinite(); dyn0.Symmetrize()

    # --- FriedelCorrection + Path-P MACE backbone ---
    fc_bg = ph.force_constants
    fc_sh = fm.load_ph(a.template).force_constants
    corr = FriedelCorrection(ph, fc_bg, fc_sh, rmin=1.0, rmax=14.0, B_law=Blaw, kappa_law=Klaw)
    ref = phonopy_to_ase(ph.supercell)
    from td_common import get_calc
    mace = get_calc("mace", a.model, device=a.device)
    supercell = dyn0.GetSupercell()

    Tel = [float(x) for x in a.tel.split(",")]
    Tlat = [float(x) for x in a.tlat.split(",")]
    out_csv = SK / f"{a.tag}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["T_el_K", "T_lat_K", "B", "soft_mode_cm", "n_imag"])
    print(f"[b2d-2d] grid T_el={Tel} x T_lat={Tlat} ({len(Tel)*len(Tlat)} SSCHA runs)", flush=True)

    for T_el in Tel:
        calc = FriedelMACECalculator(mace, ref, corr, T_el)   # fixed T_el -> fixed dfc
        dyn = dyn0
        for T_lat in Tlat:
            t0 = time.perf_counter()
            dyn.ForcePositiveDefinite(); dyn.Symmetrize()
            ens = Ensemble(dyn, T0=T_lat, supercell=supercell)
            minim = SSCHA_Minimizer(ens)
            minim.min_step_dyn = 0.05; minim.kong_liu_ratio = 0.5
            relax = SSCHA(minim, ase_calculator=calc, N_configs=a.nconfigs, max_pop=a.maxpop)
            relax.relax(get_stress=False)
            hess = relax.minim.ensemble.get_free_energy_hessian()
            wh, _ = hess.DiagonalizeSupercell()
            wcm = wh * RY_TO_CM
            minf = float(wcm.min()); n_imag = int((wcm < -1.0).sum())
            with open(out_csv, "a", newline="") as fh:
                csv.writer(fh).writerow([f"{T_el:.0f}", f"{T_lat:.0f}", f"{Blaw(T_el):.3f}", f"{minf:.2f}", n_imag])
            print(f"[b2d-2d] T_el={T_el:.0f} (B={Blaw(T_el):.3f}) T_lat={T_lat:.0f}: "
                  f"soft-mode={minf:.1f} cm^-1 (n_imag={n_imag}) [{time.perf_counter()-t0:.0f}s]", flush=True)
            dyn = relax.minim.dyn   # warm-start next T_lat
    print(f"[b2d-2d] wrote {out_csv}", flush=True)

    # --- heatmap ---
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = np.array([[float(x) for x in r] for r in csv.reader(open(out_csv))][1:])
    Z = np.full((len(Tlat), len(Tel)), np.nan)
    for row in data:
        i = Tlat.index(row[1]); j = Tel.index(row[0]); Z[i, j] = row[3]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="RdBu",
                   extent=[Tel[0], Tel[-1], Tlat[0], Tlat[-1]], vmin=-max(abs(Z.min()), 1), vmax=max(abs(Z.min()), 1))
    for row in data:
        ax.text(row[0], row[1], f"{row[3]:.0f}", ha="center", va="center", fontsize=8, color="k")
    ax.set_xlabel(r"$T_{\rm el}$ [K]  (Friedel amplitude decays -> (E) kink)"); ax.set_ylabel(r"$T_{\rm lat}$ [K]  ((L) kink)")
    ax.set_title(r"unified model  kink$(T_{\rm el},T_{\rm lat})$  [cm$^{-1}$]  (VSe$_2$)")
    fig.colorbar(im, ax=ax, label="soft-mode [cm$^{-1}$]")
    fig.tight_layout(); fig.savefig(SK / f"{a.tag}.png", dpi=150)
    print(f"[b2d-2d] wrote {SK / (a.tag+'.png')}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
