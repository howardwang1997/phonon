"""M1.2 figure: temperature-dependent graphene dispersion omega(q,T) (L-channel)
plus the T-trends of the top-optical frequency and the Kohn-kink at Gamma/K.

Left: dispersion overlay coloured by T (cool->warm). Right: omega_Gamma/omega_K and
kink_K vs T. All curves are the (L) lattice-anharmonic channel from the MLIP; the
(E) electronic Fermi-smearing of the anomaly is NOT included (needs DFPT).

    conda run -n phonon python scripts/plot_td_dispersion.py \
        --npz results/td_phonon/td_graphene_ft.npz
"""
from __future__ import annotations

import argparse
import csv as csvmod
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CM = 33.35641
LIT = {"gamma": 1600.0, "k": 1300.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/td_phonon/td_graphene_ft.npz")
    ap.add_argument("--out", default="results/figures/graphene_td_dispersion.png")
    a = ap.parse_args()

    d = dict(np.load(ROOT / a.npz, allow_pickle=True))
    temps = [int(t) for t in d["temperatures"]]
    labels = [str(x) for x in d["labels"]]
    label_pos = d["label_positions"]

    csv_path = (ROOT / a.npz).with_suffix(".csv")
    trend = {k: [] for k in ("T", "wG", "wK", "kG", "kK")}
    if csv_path.exists():
        for r in csvmod.DictReader(open(csv_path)):
            trend["T"].append(float(r["T_K"]))
            trend["wG"].append(float(r["w_gamma_cm"]))
            trend["wK"].append(float(r["w_k_cm"]))
            trend["kG"].append(float(r["kink_gamma"]))
            trend["kK"].append(float(r["kink_k"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, (axd, axf, axk) = plt.subplots(1, 3, figsize=(12.5, 3.8))

    norm = Normalize(vmin=min(temps), vmax=max(temps))
    cmap = plt.get_cmap("coolwarm")
    for T in temps:
        dist, freq = d[f"T{T}_dist"], d[f"T{T}_freq"] * CM
        c = cmap(norm(T))
        axd.plot(dist, freq[:, 0], color=c, lw=1.1, alpha=0.9, label=f"{T} K")
        axd.plot(dist, freq[:, 1:], color=c, lw=1.1, alpha=0.9)
    for x in label_pos:
        axd.axvline(x, color="0.7", lw=0.5, zorder=0)
    axd.set_xticks(label_pos)
    axd.set_xticklabels(labels)
    axd.set_xlim(d[f"T{temps[0]}_dist"][0], d[f"T{temps[0]}_dist"][-1])
    axd.set_ylabel("Frequency (cm$^{-1}$)")
    axd.set_title("graphene $\\omega$(q,T)  (L-channel)")
    axd.legend(frameon=False, fontsize=8, title="T")

    if trend["T"]:
        axf.plot(trend["T"], trend["wG"], "o-", color="#d1495b", label="$\\Gamma$-E$_{2g}$")
        axf.plot(trend["T"], trend["wK"], "s-", color="#2a9d4a", label="K-A$_1'$")
        axf.axhline(LIT["gamma"], color="0.5", ls=":", lw=0.9)
        axf.axhline(LIT["k"], color="0.5", ls=":", lw=0.9)
        axf.set_xlabel("T (K)")
        axf.set_ylabel("top-optical $\\omega$ (cm$^{-1}$)")
        axf.set_title("frequency vs T")
        axf.legend(frameon=False, fontsize=9)

        axk.plot(trend["T"], trend["kK"], "s-", color="#2a9d4a", label="K-A$_1'$ cusp")
        axk.plot(trend["T"], trend["kG"], "o-", color="#d1495b", label="$\\Gamma$ cusp")
        axk.set_xlabel("T (K)")
        axk.set_ylabel("top-branch kink |dv|")
        axk.set_title("Kohn-cusp sharpness vs T")
        axk.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))

    print(f"{'T(K)':>6s} {'wGamma':>9s} {'wK':>9s} {'kinkGamma':>10s} {'kinkK':>8s}")
    for i, T in enumerate(trend["T"]):
        print(f"{T:6.0f} {trend['wG'][i]:9.1f} {trend['wK'][i]:9.1f} "
              f"{trend['kG'][i]:10.1f} {trend['kK'][i]:8.1f}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
