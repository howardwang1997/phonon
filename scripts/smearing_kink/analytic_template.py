"""(c) Analytic Fermi-surface Friedel waveform D0_analytic(R) -- drops the one
measured template per material.  The cusp is DIRECTIONAL (nesting at q=K) and
full-tensor, so an isotropic cos(2k_F|R|) fails (verified earlier); the correct
analytic form sums over the STAR of the nesting vector and carries an in-plane
tensor structure:

    W(R) = [ sum_{K in star} cos(K . R + phi) ] * exp(-R/xi0) / R^p
    D0_analytic(R) = W(R) * ( cL n^ n^ + cT t^ t^ + cz z^ z^ )      t^ = z^ x n^

K (=2k_F nesting vector for the K-A1' anomaly) is fixed by the lattice: the C3
star of the fractional K=(1/3,1/3). We fit {phi, p, xi0} (nonlinear) + {cL,cT,cz}
(linear) to the MEASURED D0 once, then check the analytic template still
reproduces kink_K(T_el).  If yes, no per-material template measurement is needed
-- only k_F from the bands.

    conda run --no-capture-output -n phonon python scripts/smearing_kink/analytic_template.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import friedel_module as fm
from fit_friedel import DG, TEL, DFT_KINKK, YDIR, BG, REF

ZHAT = np.array([0.0, 0.0, 1.0])


def k_star(ph):
    """3 Cartesian K vectors = C3 star of fractional (1/3,1/3,0)."""
    cell = np.array(ph.primitive.cell)
    recip = 2 * np.pi * np.linalg.inv(cell).T        # rows = reciprocal vectors
    K1 = np.array([1/3, 1/3, 0.0]) @ recip
    def rotz(v, th):
        c, s = np.cos(th), np.sin(th)
        return np.array([c*v[0]-s*v[1], s*v[0]+c*v[1], v[2]])
    return [K1, rotz(K1, 2*np.pi/3), rotz(K1, 4*np.pi/3)]


def analytic_D0(tabs, Ks, phi, p, xi0, coeffs, rmin=1.0, rmax=12.0,
                subl=None, row_sub=(0, 1)):
    """Per-row list of (nsuper,3,3) analytic template blocks. coeffs = (3,) or
    (2,3) [same;cross sublattice]."""
    coeffs = np.atleast_2d(coeffs)
    two = coeffs.shape[0] == 2
    rows = []
    for r, t in enumerate(tabs):
        vec, R, n = t["vec"], t["R"], t["n"]
        blk = np.zeros((len(R), 3, 3))
        m = (R >= rmin) & (R <= rmax)
        W = np.zeros(len(R))
        for K in Ks:
            W += np.cos(vec @ K + phi)
        env = np.where(R > 0, np.exp(-R / xi0) / R ** p, 0.0)
        W = W * env
        for i in np.where(m)[0]:
            nn = np.outer(n[i], n[i])
            th = np.cross(ZHAT, n[i]); nt = np.linalg.norm(th)
            tt = np.outer(th, th) / (nt**2) if nt > 1e-9 else np.zeros((3, 3))
            zz = np.outer(ZHAT, ZHAT)
            ci = 1 if (two and subl is not None and subl[i] != row_sub[r]) else 0
            cL, cT, cz = coeffs[ci]
            blk[i] = W[i] * (cL * nn + cT * tt + cz * zz)
        rows.append(blk)
    return rows


def _sublattice(ph):
    """0/1 sublattice label per supercell atom (graphene: first half A, rest B)."""
    n = len(ph.supercell); return (np.arange(n) >= n // 2).astype(int)


def fit_tensor_coeffs(tabs, D0_meas, Ks, phi, p, xi0, rmin, rmax, subl=None,
                      row_sub=(0, 1)):
    """Linear LSQ for tensor coeffs given nonlinear shape params.  If subl is
    given, same- vs cross-sublattice pairs get INDEPENDENT tensor coeffs
    (6 total) -- captures the graphene A-A / A-B RKKY phase flip.  Returns
    (coeffs, rel_resid); coeffs shape (3,) or (2,3) [same;cross]."""
    two = subl is not None
    ncol = 6 if two else 3
    A_cols = [[] for _ in range(ncol)]
    tgt = []
    for r, t in enumerate(tabs):
        vec, R, n = t["vec"], t["R"], t["n"]
        m = (R >= rmin) & (R <= rmax)
        W = np.zeros(len(R))
        for K in Ks:
            W += np.cos(vec @ K + phi)
        env = np.where(R > 0, np.exp(-R / xi0) / R ** p, 0.0)
        W = W * env
        for i in np.where(m)[0]:
            nn = np.outer(n[i], n[i])
            th = np.cross(ZHAT, n[i]); nt = np.linalg.norm(th)
            tt = np.outer(th, th) / (nt**2) if nt > 1e-9 else np.zeros((3, 3))
            zz = np.outer(ZHAT, ZHAT)
            bases = [W[i]*nn, W[i]*tt, W[i]*zz]
            off = 3 if (two and subl[i] != row_sub[r]) else 0
            for k in range(ncol):
                A_cols[k].append(bases[k-off].ravel() if off <= k < off+3
                                 else np.zeros(9))
            tgt.append(D0_meas[r][i].ravel())
    A = np.stack([np.concatenate(c) for c in A_cols], axis=1)
    b = np.concatenate(tgt)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = np.linalg.norm(A @ coeffs - b) / (np.linalg.norm(b) + 1e-12)
    return (coeffs.reshape(2, 3) if two else coeffs), resid


def main():
    ph = fm.load_ph(YDIR / f"graphene_sc6_dg{BG}_phonopy.yaml")
    fc_bg = ph.force_constants
    fc_ref = fm.load_ph(YDIR / f"graphene_sc6_dg{REF}_phonopy.yaml").force_constants
    tabs, p2s = fm.pair_table(ph)
    D0_meas = fm.template_delta(fc_ref, fc_bg, tabs)
    Ks = k_star(ph)
    print(f"# |K| = {np.linalg.norm(Ks[0]):.3f} A^-1  (period {2*np.pi/np.linalg.norm(Ks[0]):.2f} A); "
          f"star of 3")

    rmin, rmax = 1.0, 12.0
    subl = _sublattice(ph)
    row_sub = (subl[p2s[0]], subl[p2s[1]])
    # scan nonlinear shape params (phi, p, xi0); linear tensor coeffs inside.
    # sublattice-resolved (same vs cross A/B pairs get independent tensor coeffs).
    best = None
    for p in (1, 2, 3):
        for phi in np.linspace(0, np.pi, 9, endpoint=False):
            for xi0 in (6, 9, 12, 18, 30, 60, 1e6):
                c, res = fit_tensor_coeffs(tabs, D0_meas, Ks, phi, p, xi0, rmin, rmax,
                                           subl=subl, row_sub=row_sub)
                if best is None or res < best[0]:
                    best = (res, phi, p, xi0, c)
    res, phi, p, xi0, c = best
    print(f"# analytic D0 fit to measured D0 (sublattice-resolved): rel-resid={res:.3f}  "
          f"(phi={phi:.2f}, p={p}, xi0={xi0:.0f})\n#   coeffs[same/cross cL,cT,cz]={c.round(2)}")

    D0_ana = analytic_D0(tabs, Ks, phi, p, xi0, c, rmin, rmax, subl=subl, row_sub=row_sub)

    # reproduce kink with analytic template (per-smearing B,kappa fit)
    kappas = np.linspace(0.0, 2.0, 161)
    print(f"\n{'dg':>6} {'T_el':>6} {'kink_meas':>9} {'kink_ana':>9} {'DFT':>6}")
    errs_a, errs_m = [], []
    for dg in DG:
        if dg in (BG,):
            continue
        fcd = fm.load_ph(YDIR / f"graphene_sc6_dg{dg}_phonopy.yaml").force_constants
        Bm, km, _ = fm.fit_template_env(fcd, fc_bg, tabs, D0_meas, rmin, rmax, kappas)
        Ba, ka, _ = fm.fit_template_env(fcd, fc_bg, tabs, D0_ana, rmin, rmax, kappas)
        fcm = fm.add_template(fc_bg, tabs, D0_meas, Bm, km, rmin, rmax)
        fca = fm.add_template(fc_bg, tabs, D0_ana, Ba, ka, rmin, rmax)
        kkm = fm.kink_of(ph, fcm)[0]
        kka = fm.kink_of(ph, fca)[0]
        kd = DFT_KINKK[dg]
        errs_m.append(abs(kkm-kd)); errs_a.append(abs(kka-kd))
        print(f"{dg:>6} {TEL[dg]:>6} {kkm:>9.2f} {kka:>9.2f} {kd:>6.2f}")
    print(f"# MAE vs DFT: measured-template={np.mean(errs_m):.2f}  "
          f"analytic-template={np.mean(errs_a):.2f}")


if __name__ == "__main__":
    main()
