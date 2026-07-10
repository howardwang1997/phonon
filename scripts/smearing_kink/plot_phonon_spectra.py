"""Plot phonon SPECTRA for the weekly report: MLIP+anharmonic-FT method on
(E) smearing-dependent and (L) temperature-dependent phonon dispersions.

(E): phonopy band structure (Gamma-M-K-Gamma) from DFT fc2 at several smearings (degauss).
(L): SSCHA free-energy-Hessian bands at several T_lat (from vq3e_sscha_bands.py npz).
Both overlaid on one axis per material, coloured by smearing / temperature.

Run:
  (E) conda run -n phonon python scripts/smearing_kink/plot_phonon_spectra.py E
  (L) conda run -n phonon python scripts/smearing_kink/plot_phonon_spectra.py L
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SK = ROOT / "results" / "smearing_kink"
TD = ROOT / "results" / "td_phonon"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phonopy

LABELS = ["$\\Gamma$", "M", "K", "$\\Gamma$"]
POINTS = [np.array([0., 0., 0.]), np.array([0.5, 0., 0.]),
          np.array([1/3, 1/3, 0.]), np.array([0., 0., 0.])]


def band_structure(yaml_path, nseg=40):
    """q-path Gamma-M-K-Gamma via direct run_qpoints (robust). Returns x, freq(nq,nbands), tick_x."""
    ph = phonopy.load(str(yaml_path), is_compact_fc=False)
    qs, tick_idx = [], [0]
    for i in range(len(POINTS) - 1):
        q0 = POINTS[i]; q1 = POINTS[i + 1]
        for j in range(1, nseg + 1):
            qs.append(q0 + (q1 - q0) * j / nseg)
        tick_idx.append(len(qs))
    qs = np.array(qs)
    ph.run_qpoints(qs, with_dynamical_matrices=False)
    freq_thz = np.array(ph.get_qpoints_dict()["frequencies"])  # (nq, nbands) THz (phonopy default)
    freq = freq_thz * 33.356  # -> cm^-1
    seglen = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    x = np.concatenate([[0.0], np.cumsum(seglen)])
    tick_idx_arr = np.clip(np.array(tick_idx), 0, len(x) - 1)
    tick_x = x[tick_idx_arr]
    return x, freq, tick_x


def _annot(ax, title, ylim):
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("frequency [cm$^{-1}$]")
    ax.set_ylim(*ylim)
    # vertical lines at the band-path ticks
    d = ax.get_xlim()
    # tick positions: 0, len/3, 2len/3, len approx — set by concatenation; place at segment ends
    ax.axhline(0, color="#aaa", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_E():
    """(E) smearing spectra: 4 materials, 4 degauss each (2x2 grid)."""
    mats = [
        ("1T-VSe$_2$ (1T, lattice CDW)", (-110, 310), [
            ("0.005", ROOT/"results"/"vq_family"/"vse2"/"1T-VSe2_dg0.005.yaml"),
            ("0.010", ROOT/"results"/"vq_family"/"vse2"/"1T-VSe2_dg0.010.yaml"),
            ("0.015", ROOT/"results"/"vq_family"/"vse2"/"1T-VSe2_dg0.015.yaml"),
            ("0.020", ROOT/"results"/"vq_family"/"vse2"/"1T-VSe2_dg0.020.yaml"),
        ]),
        ("NbSe$_2$ (2H, electronic CDW)", (-140, 310), [
            ("0.005", ROOT/"results"/"v100"/"fc2_nbse2_3x3_0.005"/"NbSe2_phonopy.yaml"),
            ("0.010", ROOT/"results"/"v100"/"fc2_nbse2_3x3_0.010"/"NbSe2_phonopy.yaml"),
            ("0.015", ROOT/"results"/"v100"/"fc2_nbse2_3x3_0.015"/"NbSe2_phonopy.yaml"),
            ("0.020", ROOT/"results"/"v100"/"fc2_nbse2_3x3_0.020"/"NbSe2_phonopy.yaml"),
        ]),
        ("NbS$_2$ (2H)", (-120, 310), [
            ("0.005", ROOT/"results"/"v100"/"fc2_nbs2_3x3_0.005"/"NbS2_phonopy.yaml"),
            ("0.010", ROOT/"results"/"v100"/"fc2_nbs2_3x3_0.010"/"NbS2_phonopy.yaml"),
            ("0.015", ROOT/"results"/"v100"/"fc2_nbs2_3x3_0.015"/"NbS2_phonopy.yaml"),
            ("0.020", ROOT/"results"/"v100"/"fc2_nbs2_3x3_0.020"/"NbS2_phonopy.yaml"),
        ]),
        ("2H-TaSe$_2$ (2H)", (-150, 310), [
            ("0.005", ROOT/"results"/"v100"/"fc2_tase2_3x3_0.005"/"2H-TaSe2_phonopy.yaml"),
            ("0.010", ROOT/"results"/"v100"/"fc2_tase2_3x3_0.010"/"2H-TaSe2_phonopy.yaml"),
            ("0.015", ROOT/"results"/"v100"/"fc2_tase2_3x3_0.015"/"2H-TaSe2_phonopy.yaml"),
            ("0.020", ROOT/"results"/"v100"/"fc2_tase2_3x3_0.020"/"2H-TaSe2_phonopy.yaml"),
        ]),
    ]
    cmap = plt.cm.Blues
    n = 4
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    for ax, (mat, ylim, files) in zip(axes.flat, mats):
        tick_x = None
        for i, (dg, yml) in enumerate(files):
            if not yml.exists():
                print(f"  [E] skip {mat} {dg}: {yml} missing"); continue
            x, freq, tick_x = band_structure(yml)
            color = cmap(0.30 + 0.62 * i / (n - 1))
            ax.plot(x, freq, color=color, lw=1.1, alpha=0.85, label=f"dg={dg}")
        if tick_x is not None:
            ax.set_xticks(tick_x); ax.set_xticklabels(LABELS)
            for xt in tick_x[1:-1]:
                ax.axvline(xt, color="#ccc", lw=0.6)
            ax.set_xlim(tick_x[0], tick_x[-1])
        ax.axhline(0, color="#aaa", lw=0.6)
        ax.set_title(f"(E) smearing — {mat}", fontsize=10)
        ax.set_ylim(*ylim)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("frequency [cm$^{-1}$]")
    # single shared legend OUTSIDE (below the panels) — generous bottom margin, no tight crop
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005),
               frameon=False, fontsize=10, title="degauss (Ry) — darker = sharper Fermi surface")
    fig.suptitle("(E)-channel: phonon dispersion vs electronic smearing  (Kohn soft mode melts with T$_{el}$=degauss×157888 K)",
                 fontsize=11.5, y=0.99)
    fig.subplots_adjust(left=0.07, bottom=0.17, right=0.97, top=0.93, hspace=0.30, wspace=0.13)
    out = SK / "spectra_E_smearing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)


def plot_L():
    """(L) temperature spectra: VSe2 + NbSe2, 4 T_lat each (from SSCHA-bands npz)."""
    mats = {
        "1T-VSe$_2$ (lattice-driven, T$_{CDW}$=110K)": ("VSe2_Lband", (-700, 350)),
        "NbSe$_2$ (2H, electronic)": ("NbSe2_Lband", (-200, 350)),
    }
    Ts = [20, 100, 200, 300]
    cmap = plt.cm.Reds
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (mat, (tag, ylim)) in zip(axes, mats.items()):
        npz = TD / f"{tag}.npz"
        if not npz.exists():
            print(f"  [L] skip {mat}: {npz} missing (R3 not done?)"); continue
        d = np.load(npz, allow_pickle=True)
        x = d["x"]; ticks = d["ticks"]; labels = d["labels"]
        for i, T in enumerate(Ts):
            key = f"T{T}_bands"
            if key not in d:
                print(f"  [L] {mat}: {key} missing"); continue
            bands = d[key]  # (nq, nbands) cm^-1
            color = cmap(0.35 + 0.6 * i / (len(Ts) - 1))
            ax.plot(x, bands, color=color, lw=1.2, alpha=0.85, label=f"T$_{{lat}}$={T} K")
        ax.set_xticks(ticks); ax.set_xticklabels([str(l) for l in labels])
        for t in ticks[1:-1]:
            ax.axvline(t, color="#ccc", lw=0.6)
        ax.axhline(0, color="#aaa", lw=0.6)
        ax.set_title(f"(L) temperature spectra — {mat}", fontsize=11)
        ax.set_ylabel("frequency [cm$^{-1}$]"); ax.set_ylim(*ylim)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_xlim(x[0], x[-1])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("(L)-channel: SSCHA phonon dispersion vs lattice temperature (Path-P anharmonic backbone)", fontsize=12)
    fig.tight_layout()
    out = SK / "spectra_L_temperature.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)


def plot_L_tdep():
    """(L) temperature FULL spectrum from TDEP effective fc2(T) — real-freq dispersion at each T.
    VSe2 (1T), M-Gamma-K-M path, 6 T_lat. Complements the SSCHA soft-mode curve (which captures
    the imaginary crossover that TDEP, giving real freqs, cannot)."""
    npz = TD / "td_1T-VSe2_tdep.npz"
    if not npz.exists():
        print(f"  [Ltdep] skip: {npz} missing"); return
    d = np.load(npz, allow_pickle=True)
    Ts = [50, 80, 110, 150, 200, 300]
    cmap = plt.cm.Reds
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ticks = d["label_positions"]; labels = [str(l) for l in d["labels"]]
    for i, T in enumerate(Ts):
        if f"T{T}_freq" not in d:
            continue
        dist = d[f"T{T}_dist"]; freq = d[f"T{T}_freq"] * 33.356  # THz -> cm^-1
        ax.plot(dist, freq, color=cmap(0.30 + 0.62 * i / (len(Ts) - 1)), lw=1.2, alpha=0.8,
                label=f"{T} K")
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    for t in ticks[1:-1]:
        ax.axvline(t, color="#ccc", lw=0.6)
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.set_ylabel("frequency [cm$^{-1}$]"); ax.set_ylim(-15, 360)
    ax.set_xlim(ticks[0], ticks[-1])
    # legend OUTSIDE (bottom, horizontal) so it never covers the bands
    ax.legend(title="$T_{lat}$", loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=6,
              frameon=False, fontsize=8.5, title_fontsize=9)
    ax.set_title("(L) temperature spectrum — 1T-VSe$_2$  (TDEP effective fc$_2$(T), real freqs)", fontsize=10.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.10, bottom=0.20, right=0.97, top=0.92)
    out = SK / "spectra_L_temperature.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)


def plot_kink_comparison():
    """Kohn-anomaly kink (soft-mode depth at CDW q) vs T_el (smearing) AND vs T_lat (temperature).
    Left: (E) kink melts with electronic smearing. Right: (L) kink heals with lattice temperature
    (at T_CDW). The two channels of the kink(T_el, T_lat) variation."""
    import csv
    fig, (axE, axL) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    # ---- (E): |soft-mode| vs T_el (smearing), from family_melting.csv ----
    melt_path = SK / "family_melting.csv"
    if melt_path.exists():
        melt = list(csv.DictReader(open(melt_path)))
        for fmat, lab, col in [("1T-VSe2", "1T-VSe$_2$", "#c0504d"), ("NbSe2", "NbSe$_2$", "#2f6f9f"),
                               ("2H-TaS2", "2H-TaS$_2$", "#2f9f6f"), ("2H-TaSe2", "2H-TaSe$_2$", "#9f6f2f")]:
            rows = sorted([r for r in melt if r["material"] == fmat], key=lambda r: float(r["T_el_K"]))
            if not rows:
                continue
            T = [float(r["T_el_K"]) for r in rows]
            d = [abs(float(r["minfreq_THz"])) for r in rows]
            axE.plot(T, d, "-o", color=col, lw=1.8, ms=6, label=lab)
    axE.set_xlabel("electronic smearing  $T_{el}$ [K]  (degauss$\\times$157888)", fontsize=10)
    axE.set_ylabel("|Kohn soft-mode (kink depth)|  [THz]", fontsize=10)
    axE.set_title("(E): kink melts with smearing", fontsize=11)
    axE.axhline(0, color="#aaa", lw=0.6)
    for s in ("top", "right"):
        axE.spines[s].set_visible(False)

    # ---- (L): |soft-mode| vs T_lat (temperature), from SSCHA csv ----
    CM2THZ = 1.0 / 33.356
    for mat, files, col, Tcdw in [("1T-VSe$_2$", ["1T-VSe2_fine.csv", "1T-VSe2_L.csv"], "#c0504d", 110),
                                   ("1T-TiSe$_2$", ["1T-TiSe2_L.csv"], "#e08a2e", 200)]:
        pts = []
        for fn in files:
            p = TD / fn
            if not p.exists():
                continue
            pts += [(float(r["T_K"]), abs(float(r["sscha_minfreq_cm"])) * CM2THZ)
                    for r in csv.DictReader(open(p))]
        pts = sorted(set(pts))
        if pts:
            axL.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col, lw=1.8, ms=6, label=mat)
            axL.axvline(Tcdw, color=col, ls=":", lw=1.0, alpha=0.6)
            axL.text(Tcdw, axL.get_ylim()[1] * 0.85 if axL.get_ylim()[1] > 0 else 8,
                     f"$T_{{CDW}}$={Tcdw}K", color=col, fontsize=8, ha="center")
    axL.set_xlabel("lattice temperature  $T_{lat}$ [K]", fontsize=10)
    axL.set_ylabel("|Kohn soft-mode (kink depth)|  [THz]", fontsize=10)
    axL.set_title("(L): kink heals with temperature (at $T_{CDW}$)", fontsize=11)
    axL.set_xlim(0, 320)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    # shared legend OUTSIDE (top) for both panels
    handles, labels = axE.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0),
               frameon=False, fontsize=9)
    fig.suptitle("Kohn-anomaly kink depth vs smearing ($T_{el}$) and temperature ($T_{lat}$) — the two channels of $kink(T_{el},T_{lat})$",
                 fontsize=11, y=0.95)
    fig.subplots_adjust(left=0.08, bottom=0.13, right=0.97, top=0.80, wspace=0.22)
    out = SK / "kink_vs_Tel_Tlat.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)


def plot_deploy_vs_dft(mat="NbSe2", label=None, ylim=None):
    """MLIP+long-range vs DFT full spectrum (does the method reproduce DFT?).
    deployed (backbone + Friedel) vs DFT vs smearing-blind backbone."""
    npz = SK / f"deploy_vs_dft_{mat}.npz"
    if not npz.exists():
        print(f"  [deploy] skip: {npz} missing"); return
    d = np.load(npz, allow_pickle=True)
    qs = d["qs"]
    seglen = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    x = np.concatenate([[0.0], np.cumsum(seglen)])
    seg_len = len(x) // 3
    tick_x = [x[0], x[seg_len - 1], x[2 * seg_len - 1], x[-1]]
    Ts = [int(t) for t in d["tel"]]
    lab = label or mat
    fig, axes = plt.subplots(1, len(Ts), figsize=(7.2 * len(Ts), 5.5), sharey=True)
    if len(Ts) == 1:
        axes = [axes]
    for ax, T in zip(axes, Ts):
        ax.plot(x, d[f"dft_T{T}"], color="#222", lw=2.0, alpha=0.9, label="DFT (target)")
        ax.plot(x, d[f"mlip_T{T}"], color="#c0504d", lw=1.5, ls="--", alpha=0.95,
                label="MLIP + long-range Friedel")
        ax.plot(x, d["backbone"], color="#2f6f9f", lw=1.0, alpha=0.5,
                label="backbone (smearing-blind)")
        ax.set_xticks(tick_x); ax.set_xticklabels(LABELS)
        for xt in tick_x[1:-1]:
            ax.axvline(xt, color="#ccc", lw=0.6)
        ax.axhline(0, color="#aaa", lw=0.6)
        mae = np.mean(np.abs(d[f"mlip_T{T}"] - d[f"dft_T{T}"]))
        ax.set_title(f"{lab}  $T_{{el}}$={T} K  (full-band MAE = {mae:.2f} cm$^{{-1}}$)", fontsize=10.5)
        ax.set_xlim(tick_x[0], tick_x[-1])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # auto ylim from data (no cropping); or use provided
    if ylim is None:
        allf = np.concatenate([d[f"dft_T{T}"].ravel() for T in Ts] + [d["backbone"].ravel()])
        ylim = (min(allf.min() * 1.1, -10), allf.max() * 1.1 + 5)
    axes[0].set_ylim(*ylim)
    axes[0].set_ylabel("frequency [cm$^{-1}$]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005),
               frameon=False, fontsize=9.5)
    fig.suptitle(f"MLIP + long-range fine-tuning vs DFT — {lab}", fontsize=11.5, y=0.99)
    fig.subplots_adjust(left=0.07, bottom=0.18, right=0.98, top=0.93, wspace=0.10)
    out = SK / f"deploy_vs_dft_{mat}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)


def plot_graphene_L():
    """Graphene (L): DFT-MD-TDEP (DFT baseline) vs MLIP-TDEP (effective fc2), M-Gamma-K-M.
    Shows graphene is (L)-stable (no CDW soft mode) and MLIP reproduces DFT."""
    dft = np.load(TD / "graphene_dft_tdep.npz", allow_pickle=True)
    mlp = np.load(TD / "td_graphene_ft.npz", allow_pickle=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    # DFT 300K
    dfreq = dft["T300_freq"] * 33.356  # THz->cm^-1 (tdp freqs are THz)
    ddist = dft["T300_dist"]
    ax.plot(ddist, dfreq, color="#222", lw=2.2, alpha=0.9, label="DFT-MD-TDEP (300K)")
    # MLIP 300K + 100K
    for T, col, ls in [(300, "#c0504d", "--"), (100, "#2f6f9f", ":")]:
        if f"T{T}_freq" in mlp:
            ax.plot(mlp[f"T{T}_dist"], mlp[f"T{T}_freq"] * 33.356, color=col, lw=1.5, ls=ls,
                    alpha=0.9, label=f"MLIP-TDEP ({T}K)")
    ticks = dft["label_positions"]; labels = [str(l) for l in dft["labels"]]
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    for t in ticks[1:-1]:
        ax.axvline(t, color="#ccc", lw=0.6)
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.set_ylabel("frequency [cm$^{-1}$]"); ax.set_ylim(-15, 1650)
    ax.set_title("graphene (L): DFT-MD-TDEP vs MLIP — (L)-stable (no CDW soft mode)", fontsize=10.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.10, bottom=0.12, right=0.72, top=0.92)
    out = SK / "graphene_L_dft_vs_mlip.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "E"
    if which in ("E", "both"): plot_E()
    if which in ("L", "both"): plot_L()
    if which in ("Ltdep", "all"): plot_L_tdep()
    if which in ("kink", "all"): plot_kink_comparison()
    if which in ("deploy", "all"):
        for mat, lab in [("NbSe2", "NbSe$_2$ (2H)"), ("2H-TaSe2", "2H-TaSe$_2$ (2H)"),
                         ("2H-TaS2", "2H-TaS$_2$ (2H)"), ("NbS2", "NbS$_2$ (2H)"),
                         ("1T-VSe2", "1T-VSe$_2$ (1T, sharp-melting)")]:
            plot_deploy_vs_dft(mat, lab)
    if which in ("grL", "all"): plot_graphene_L()
