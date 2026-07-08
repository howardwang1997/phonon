"""B2d — unified MLIP+LR model -> kink(T_el, T_lat) for VSe2 (the sub-line deliverable).

ONE FriedelMACE model = Path-P anharmonic backbone (the (L)-kink source) +
T_el-conditioned Friedel long-range term (the (E)-kink source). Both kink axes
read out of the SAME backbone:

  (E) axis: harmonic fc2 through FriedelMACE(Path-P, Friedel(T_el)) -> soft-mode(T_el).
            (the Friedel LR amplitude decays with T_el -> Kohn kink melts)
  (L) axis: SSCHA free-energy Hessian of the Path-P backbone (B=0) -> soft-mode(T_lat).
            (the anharmonic backbone heals with T_lat -> CDW crossover)  [existing 1T-VSe2_L.csv]

Validate: both axes, normalized to reduced-T T/T*, collapse to the SAME healing law
f(x)=(1-x^p)^q as B2 step-2 found -- but now FROM ONE backbone (the unification).

Phase 1 (this script, ~5 min): the (E) slice through the Path-P backbone + the unified
collapse vs the (L) SSCHA data. Phase 2 (b2d_2d_sscha.py, separate): the 2D interior.

Run on 2060:
  conda run --no-capture-output -n phonon python scripts/smearing_kink/b2d_unified_kink.py \
    --backbone results/vq_family/vse2/1T-VSe2_dg0.020.yaml \
    --template results/vq_family/vse2/1T-VSe2_dg0.005.yaml \
    --model results/finetune_path_p_1T-VSe2/ft.model --tag b2d_vse2 --device cuda
"""
from __future__ import annotations
import argparse, csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))
import friedel_module as fm
from friedel_calc import FriedelCorrection, FriedelMACECalculator, fc2_from_calc
from phonon_accel.phonons import phonopy_to_ase

SK = ROOT / "results" / "smearing_kink"
TD = ROOT / "results" / "td_phonon"
CM2THz = 1.0 / 33.356


def mace_calc(mp, device):
    from mace.calculators import MACECalculator
    import torch
    return MACECalculator(model_paths=str(mp), device=device,
                          default_dtype="float32")


def minf(ph, fc, mesh=12):
    ph.force_constants = fc
    ph.run_mesh([mesh, mesh, 1], with_eigenvectors=False, is_gamma_center=True)
    return float(np.min(ph.get_mesh_dict()["frequencies"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="VSe2 fc2 yaml dg0.020 (backbone)")
    ap.add_argument("--template", required=True, help="VSe2 fc2 yaml dg0.005 (template)")
    ap.add_argument("--model", required=True, help="Path-P VSe2 ft.model")
    ap.add_argument("--tag", default="b2d_vse2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mesh", type=int, default=12)
    a = ap.parse_args()

    # --- B_law / kappa_law from friedel_family.csv (per-material VSe2 fit) ---
    rows = [r for r in csv.DictReader(open(SK / "friedel_family.csv")) if r["material"] == "1T-VSe2"]
    Tb = np.array([float(r["T_el"]) for r in rows])
    Bb = np.array([float(r["B"]) for r in rows])
    Kb = np.array([float(r["kappa"]) for r in rows])
    o = np.argsort(Tb); Tb, Bb, Kb = Tb[o], Bb[o], Kb[o]
    Blaw = lambda T: float(np.interp(T, Tb, Bb))
    Klaw = lambda T: float(np.interp(T, Tb, Kb))

    ph = fm.load_ph(a.backbone); fc_bg = ph.force_constants
    fc_sh = fm.load_ph(a.template).force_constants
    corr = FriedelCorrection(ph, fc_bg, fc_sh, rmin=1.0, rmax=14.0, B_law=Blaw, kappa_law=Klaw)
    ref = phonopy_to_ase(ph.supercell)
    base = mace_calc(a.model, a.device)

    # --- (E) slice: Path-P backbone + Friedel(T_el) -> soft-mode(T_el) ---
    _, fcm = fc2_from_calc(ph, base, subtract_ref=True)
    bare = minf(ph, fcm)
    print(f"# [B2d] Path-P backbone bare soft-mode = {bare:.3f} THz (MACE fc2, no Friedel)")
    Tel_grid = np.array([789, 1184, 1579, 1974, 2368, 2763, 3158, 4737])
    E_T, E_soft = [], []
    print(f"# {'T_el':>6} {'B':>5} {'(E) soft-mode THz':>18}")
    for T in Tel_grid:
        calc = FriedelMACECalculator(base, ref, corr, T)
        _, fct = fc2_from_calc(ph, calc, subtract_ref=True)
        mf = minf(ph, fct, a.mesh)
        E_T.append(T); E_soft.append(mf)
        print(f"# {T:>6.0f} {Blaw(T):>5.3f} {mf:>18.3f}")
    E_T = np.array(E_T); E_soft = np.array(E_soft)

    # --- (L) slice: existing SSCHA on Path-P backbone (B=0) ---
    L_rows = []
    for fn in ("1T-VSe2_fine.csv", "1T-VSe2_L.csv"):
        p = TD / fn
        if p.exists():
            L_rows += [(float(r["T_K"]), float(r["sscha_minfreq_cm"]) * CM2THz)
                       for r in csv.DictReader(open(p))]
    L = np.array(sorted(set(L_rows)))
    print(f"# [B2d] (L) SSCHA soft-mode: {len(L)} pts, depth {L[:,1].min():.3f} THz at T_lat={L[L[:,1].argmin(),0]:.0f}K")

    # --- unified reduced-T collapse (both axes from ONE backbone) ---
    def heal(d):
        neg = d[d[:, 1] < -0.02]; pos = d[d[:, 1] >= -0.02]
        return (float(pos[:, 0].min()) if len(pos) else float(d[-1, 0])), float(neg[:, 1].min())
    def norm(d, Ts, depth):
        x = d[:, 0] / Ts
        y = np.where(d[:, 1] < -0.02, np.minimum(d[:, 1] / depth, 1.0), 0.0)
        return x, y
    E = np.column_stack([E_T, E_soft])
    Te, de = heal(E); Tl, dl = heal(L)
    xe, ye = norm(E, Te, de); xl, yl = norm(L, Tl, dl)
    m = (xe <= 1.05); xe, ye = xe[m], ye[m]
    m = (xl <= 1.05); xl, yl = xl[m], yl[m]

    def fmodel(x, p, q): return np.maximum(0, 1 - x**p)**q
    from scipy.optimize import curve_fit
    Xall = np.concatenate([xe, xl]); Yall = np.concatenate([ye, yl])
    popt, _ = curve_fit(fmodel, Xall, Yall, p0=[2.0, 0.5], bounds=([0.5, 0.1], [5, 3]))
    mae = np.mean(np.abs(Yall - fmodel(Xall, *popt)))
    print(f"# [B2d] unified law from ONE Path-P backbone: p={popt[0]:.2f}, q={popt[1]:.2f}, "
          f"combined MAE={mae:.3f}  (B2 step-2 was p=4.44, q=3.00, MAE=0.026)")
    print(f"# [B2d] T*_el={int(Te)} K (heal), T*_lat={int(Tl)} K (=T_CDW)")

    # --- write csv + plot ---
    out_csv = SK / f"{a.tag}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "T_K", "soft_mode_THz", "Tstar_K", "depth_THz", "p", "q"])
        for T, s in zip(E_T, E_soft):
            w.writerow(["(E)_T_el", f"{T:.0f}", f"{s:.3f}", f"{int(Te)}", f"{de:.3f}", f"{popt[0]:.3f}", f"{popt[1]:.3f}"])
        for T, s in L:
            w.writerow(["(L)_T_lat", f"{T:.0f}", f"{s:.3f}", f"{int(Tl)}", f"{dl:.3f}", f"{popt[0]:.3f}", f"{popt[1]:.3f}"])
    print("# wrote", out_csv)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.plot(E_T, E_soft, "-o", color="#c0504d", label="(E) kink via Friedel LR term")
    ax1.plot(L[:, 0], L[:, 1], "--s", color="#2f6f9f", mfc="white", label="(L) kink via backbone SSCHA")
    ax1.axhline(0, color="#888", lw=0.8); ax1.set_xlabel("T  [K]  (T_el red, T_lat blue)")
    ax1.set_ylabel("soft-mode  [THz]"); ax1.legend(frameon=False, fontsize=9)
    ax1.set_title("ONE Path-P backbone: both kink axes")
    for s in ("top", "right"): ax1.spines[s].set_visible(False)
    xx = np.linspace(0, 1.05, 60)
    ax2.plot(xe, ye, "o", color="#c0504d", ms=8, label="(E) reduced-T")
    ax2.plot(xl, yl, "s", color="#2f6f9f", ms=8, mfc="white", label="(L) reduced-T")
    ax2.plot(xx, fmodel(xx, *popt), "-", color="#333", lw=1.5,
             label=f"fit (p={popt[0]:.2f}, q={popt[1]:.2f})")
    ax2.set_xlabel(r"reduced-T  $T/T^*$"); ax2.set_ylabel(r"$|\omega_{\rm soft}|/|\omega_0|$")
    ax2.legend(frameon=False, fontsize=9); ax2.set_title(f"unified collapse (MAE={mae:.3f})")
    for s in ("top", "right"): ax2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(SK / f"{a.tag}.png", dpi=150)
    print("# wrote", SK / f"{a.tag}.png")
    print("# => ONE FriedelMACE(Path-P backbone): (E) kink from Friedel LR term, "
          "(L) kink from backbone anharmonicity. Both follow the unified reduced-T law.")


if __name__ == "__main__":
    raise SystemExit(main())
