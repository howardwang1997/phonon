"""Transfer test: does the thermal Friedel damping law kappa(T_el) fitted at the
training lattice constant a=2.46 predict kink_K(T_el) at HELD-OUT lattice
constants a=2.44/2.48/2.50?

Protocol (per held-out a): measure only 2 DFT points -- the template (sharpest
smearing) and the backbone (most-smeared) -- then predict every intermediate
kink with the a=2.46 damping law and ZERO new fitting.  This is the
few-shot-across-materials claim: the Fermi-surface waveform D0 is re-measured
once at each a; the damping *physics* transfers.

    conda run --no-capture-output -n phonon python scripts/smearing_kink/transfer_friedel.py
"""
from __future__ import annotations
import sys, csv, warnings, glob, re
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import friedel_module as fm

ROOT = Path(__file__).resolve().parents[2]
VQ = ROOT / "results" / "vq_kink6"       # a=2.46 training set
SURF = ROOT / "results" / "vq_surface"   # held-out a
OUT = ROOT / "results" / "smearing_kink"
TEL_PER_DG = 157888.0  # T_el = degauss * this


def dg_to_T(dg):  # dg string -> T_el
    return float(dg) * TEL_PER_DG


def load_set(paths):
    """paths: dict dg-> yaml. Return phs, fcs, sorted dg list (by T_el asc)."""
    phs, fcs = {}, {}
    for dg, p in paths.items():
        if not Path(p).exists() or Path(p).stat().st_size < 1_300_000:
            continue  # skip missing / still-transferring (full file is 1306912 B)
        try:
            ph = fm.load_ph(p)
        except Exception:
            continue  # corrupt/partial yaml
        phs[dg], fcs[dg] = ph, ph.force_constants
    dgs = sorted(phs, key=lambda d: float(d))
    return phs, fcs, dgs


def kinks_dft(phs, fcs, dgs):
    return {dg: fm.kink_of(phs[dg], fcs[dg])[0] for dg in dgs}


def fit_law(fcs, phs, dgs, kappas):
    """Fit template-envelope per intermediate smearing; return kappa(T) power law
    (a_pow, b) and mean B, using backbone=dgs[-1], template=dgs[0]."""
    bg, ref = dgs[-1], dgs[0]
    tabs, _ = fm.pair_table(phs[bg])
    D0 = fm.template_delta(fcs[ref], fcs[bg], tabs)
    Ts, Ks, Bs = [], [], []
    for dg in dgs[1:-1]:
        B, kap, _ = fm.fit_template_env(fcs[dg], fcs[bg], tabs, D0, 1.0, 12.0, kappas)
        Ts.append(dg_to_T(dg)); Ks.append(kap); Bs.append(B)
    Ts, Ks, Bs = map(np.array, (Ts, Ks, Bs))
    m = Ks > 1e-6
    b, loga = np.polyfit(np.log(Ts[m]), np.log(Ks[m]), 1)
    return np.exp(loga), b, Bs.mean(), (Ts, Ks, Bs)


def predict(phs, fcs, dgs, a_pow, b, Bconst, kappas):
    """Predict all kinks at this a using the given (a_pow,b,Bconst) damping law,
    with this a's OWN template/backbone. Return dict dg->predicted kink."""
    bg, ref = dgs[-1], dgs[0]
    ph = phs[bg]
    tabs, _ = fm.pair_table(ph)
    D0 = fm.template_delta(fcs[ref], fcs[bg], tabs)
    out = {}
    for dg in dgs:
        if dg == bg:
            out[dg] = fm.kink_of(ph, fcs[bg])[0]; continue
        if dg == ref:
            out[dg] = fm.kink_of(ph, fcs[ref])[0]; continue
        kap = a_pow * dg_to_T(dg) ** b
        fc = fm.add_template(fcs[bg], tabs, D0, Bconst, kap, 1.0, 12.0)
        out[dg] = fm.kink_of(ph, fc)[0]
    return out


def collect_a(prefix, directory):
    paths = {}
    for p in glob.glob(str(directory / f"{prefix}*_phonopy.yaml")):
        m = re.search(r"dg([0-9.]+)_phonopy", p)
        if m:
            paths[m.group(1)] = p
    return paths


def main():
    kappas = np.linspace(0.0, 2.0, 161)

    # --- training law at a=2.46 (restricted to smearings present in the surface)
    surf_a = {}
    for a in ("2.44", "2.48", "2.50"):
        pa = collect_a(f"gr_a{a}_", SURF)
        if pa:
            surf_a[a] = pa
    # common smearing set = whatever the surface a's have (they used 0.002..0.040)
    train_paths = {dg: str(VQ / f"graphene_sc6_dg{dg}_phonopy.yaml")
                   for dg in ["0.002", "0.005", "0.010", "0.020", "0.040"]}
    phs46, fcs46, dg46 = load_set(train_paths)
    a_pow, b, Bc, _ = fit_law(fcs46, phs46, dg46, kappas)
    print(f"# TRAIN a=2.46 (dg {dg46}): kappa(T)={a_pow:.3e}*T^{b:.2f}  B~{Bc:.3f}")

    rows = []
    # sanity: predict a=2.46 with its own law (self-consistency)
    for a, paths in [("2.46", train_paths)] + list(surf_a.items()):
        phs, fcs, dgs = load_set(paths)
        if len(dgs) < 3:
            print(f"# a={a}: only {len(dgs)} smearings ({dgs}) -- skip")
            continue
        kdft = kinks_dft(phs, fcs, dgs)
        kpred = predict(phs, fcs, dgs, a_pow, b, Bc, kappas)
        print(f"\n# a={a}  (template=dg{dgs[0]}, backbone=dg{dgs[-1]}; "
              f"predict {dgs[1:-1]} with a=2.46 law)")
        print(f"{'dg':>6} {'T_el':>6} {'kink_pred':>9} {'kink_DFT':>9} {'err':>6}")
        errs = []
        for dg in dgs:
            T = dg_to_T(dg)
            held = dg not in (dgs[0], dgs[-1])
            e = kpred[dg] - kdft[dg]
            if held:
                errs.append(abs(e))
            tag = "" if held else " (anchor)"
            print(f"{dg:>6} {T:>6.0f} {kpred[dg]:>9.2f} {kdft[dg]:>9.2f} {e:>6.2f}{tag}")
            rows.append([a, dg, T, kpred[dg], kdft[dg], int(held)])
        if errs:
            print(f"# a={a} transfer MAE on held-out smearings = {np.mean(errs):.2f} cm^-1")

    with open(OUT / "friedel_transfer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "degauss", "T_el", "kink_pred", "kink_dft", "held_out"])
        w.writerows(rows)
    print(f"\n# wrote {OUT/'friedel_transfer.csv'}")


if __name__ == "__main__":
    main()
