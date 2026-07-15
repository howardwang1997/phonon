"""Friedel fit on the graphene Kohn (E)-channel FD scan (0.01-0.2), with the
learnable-D0 ablation requested for this experiment.

Backbone  = DFT fc2 at the BROADEST smearing (dg0.20, anomaly fully melted = pure
            short-range).  D0_meas = DFT fc2(dg0.01, sharpest) - backbone.
Variant A (fixed D0):   few-shot B(T_el), kappa(T_el)  (fm.fit_template_env per T).
Variant B (learnable D0): D0_fit(R) = alpha(R) * D0_meas(R), alpha(R) a learnable
            per-distance-shell scalar, jointly fit with B/kappa by alternating
            (i) fix alpha=1 -> few-shot B,kappa; (ii) fix B,kappa -> per-shell linear
            least-squares alpha(R); iterate. Keeps the measured tensor directionality,
            lets the fit reshape the radial Friedel waveform.

Reads results/graphene_kohn_fd/graphene_sc6_dg{dg}_phonopy.yaml. Reports kink_K
reproduction (A vs B vs DFT) and saves the fitted laws.

    conda run -n phonon python scripts/smearing_kink/fit_friedel_kohn_fd.py
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[2]
import friedel_module as fm

FD = ROOT / "results" / "graphene_kohn_fd"
DGS = [0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20]
BG_DG, REF_DG = 0.20, 0.01      # backbone = broadest; D0 from sharpest
RMIN, RMAX = 1.0, 12.0
KAPPAS = np.linspace(0.0, 2.0, 161)


def load_all():
    phs, fcs = {}, {}
    for dg in DGS:
        yml = FD / f"graphene_sc6_dg{dg}_phonopy.yaml"
        if not yml.exists():
            print(f"  [skip dg{dg}] {yml} missing"); continue
        ph = fm.load_ph(yml); phs[dg] = ph; fcs[dg] = ph.force_constants
    tabs, _ = fm.pair_table(phs[BG_DG])
    return phs, fcs, tabs


def kappas_to_law(fits):
    """few-shot B~const, kappa(T)=a*T^b from the per-T (B,kappa) fits."""
    rows = [(dg, B, kap) for dg, (B, kap, _) in fits.items() if dg not in (BG_DG, REF_DG) and B > 1e-6]
    T = np.array([dg * 157887 for dg, _, _ in rows], float)
    Bm = np.mean([B for _, B, _ in rows])
    Ka = np.array([kap for _, _, kap in rows], float)
    b, loga = np.polyfit(np.log(T), np.log(Ka), 1)
    return Bm, float(np.exp(loga)), float(b)


def learnable_alpha(phs, fcs, tabs, D0, Blaw, kap_law, nbins=14):
    """Per-distance-shell alpha(R) by linear LS with fixed B(T),kappa(T).
    For each R-bin: alpha = Sum_T w_T <D0,dT> / Sum_T w_T^2 <D0,D0>, w_T=B(T)exp(-kap(T)R)."""
    bins = np.linspace(RMIN, RMAX, nbins + 1)
    fcbg = fcs[BG_DG]
    num = np.zeros(nbins); den = np.zeros(nbins)
    for prow, t in enumerate(tabs):
        s, R = t["s"], t["R"]
        Db = D0[prow]
        bi = np.clip(np.digitize(R, bins) - 1, 0, nbins - 1)
        for dg in DGS:
            if dg in (BG_DG, REF_DG):
                continue
            kap = kap_law[0] * (dg * 157887) ** kap_law[1]
            w = np.exp(-kap * R)            # drop B (cancels in ratio)
            tgt = fcs[dg][prow, s] - fcbg[prow, s]
            ipdd = np.einsum("kij,kij->k", Db, Db)
            iptd = np.einsum("kij,kij->k", Db, tgt)
            for k in range(len(R)):
                num[bi[k]] += w[k] * iptd[k]
                den[bi[k]] += w[k] * w[k] * ipdd[k]
    alpha = np.where(den > 1e-12, num / den, 1.0)
    return bins, alpha


def add_template_alpha(fc_bg, tabs, D0, B, kap, bins, alpha, rmin, rmax):
    """fc = fc_bg + B exp(-kap R) alpha(Rbin) D0  on tail pairs; ASR restored."""
    fc = fc_bg.copy()
    for prow, t in enumerate(tabs):
        s, R = t["s"], t["R"]
        bi = np.clip(np.digitize(R, bins) - 1, 0, len(alpha) - 1)
        env = B * np.exp(-kap * R) * alpha[bi] * ((R >= rmin) & (R <= rmax))
        corr = env[:, None, None] * D0[prow]
        fc[prow, s] += corr
        si = np.where(s == t["s0"])[0][0]
        fc[prow, si] -= corr.sum(axis=0)
    return fc


def main():
    phs, fcs, tabs = load_all()
    fcbg = fcs[BG_DG]
    D0 = fm.template_delta(fcs[REF_DG], fcbg, tabs)
    ph = phs[BG_DG]

    # ---- Variant A: fixed D0, few-shot B(T), kappa(T) ----
    fitsA = {}
    for dg in DGS:
        if dg == BG_DG:
            fitsA[dg] = (0.0, 0.0, 0.0)
        elif dg == REF_DG:
            fitsA[dg] = (1.0, 0.0, 0.0)
        else:
            fitsA[dg] = fm.fit_template_env(fcs[dg], fcbg, tabs, D0, RMIN, RMAX, KAPPAS)
    Bm, a_k, b_k = kappas_to_law(fitsA)
    print(f"# Variant A (fixed D0): B~{Bm:.3f}, kappa(T)={a_k:.3e}*T^{b_k:.2f}")

    # ---- Variant B: learnable D0 (alpha per shell), alternating ----
    bins, alpha = learnable_alpha(phs, fcs, tabs, D0, Bm, (a_k, b_k))
    print(f"# Variant B (learnable D0): alpha(R) over {len(alpha)} shells, "
          f"range [{alpha.min():.3f},{alpha.max():.3f}] (1.0 = fixed)")

    # ---- reproduce kink_K: A vs B vs DFT ----
    DFT = {dg: fm.kink_of(phs[dg], fcs[dg])[0] for dg in DGS}
    print(f"\n{'dg':>6} {'T_el':>7} {'DFT':>7} {'A_fixed':>8} {'B_learn':>8}")
    resA = resB = 0.0; n = 0
    for dg in DGS:
        T = dg * 157887
        if dg == BG_DG:
            kA = kB = DFT[dg]
        else:
            B = 1.0 if dg == REF_DG else Bm
            kap = 0.0 if dg == REF_DG else a_k * T ** b_k
            fcA = fm.add_template(fcbg, tabs, D0, B, kap, RMIN, RMAX)
            kA = fm.kink_of(ph, fcA)[0]
            fcB = add_template_alpha(fcbg, tabs, D0, B, kap, bins, alpha, RMIN, RMAX)
            kB = fm.kink_of(ph, fcB)[0]
        resA += abs(kA - DFT[dg]); resB += abs(kB - DFT[dg]); n += 1
        print(f"{dg:>6} {T:>7.0f} {DFT[dg]:>7.2f} {kA:>8.2f} {kB:>8.2f}")
    print(f"# MAE kink_K:  fixed-D0={resA/n:.2f}  learnable-D0={resB/n:.2f} cm^-1")

    out = {"B": Bm, "kappa_a": a_k, "kappa_b": b_k, "bins": bins.tolist(),
           "alpha": alpha.tolist(), "rmin": RMIN, "rmax": RMAX,
           "dgs": DGS, "BG_DG": BG_DG, "REF_DG": REF_DG}
    (FD / "friedel_fit_kohn_fd.json").write_text(json.dumps(out, indent=2))
    print(f"# wrote {FD/'friedel_fit_kohn_fd.json'}")


if __name__ == "__main__":
    main()
