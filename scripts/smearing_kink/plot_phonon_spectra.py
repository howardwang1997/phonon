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
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
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
        ax.legend(fontsize=7, loc="upper right", frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("frequency [cm$^{-1}$]")
    fig.suptitle("(E)-channel: phonon dispersion vs electronic smearing  (Kohn soft mode melts with T$_{el}$=degauss×157888 K)",
                 fontsize=11.5)
    fig.tight_layout()
    out = SK / "spectra_E_smearing.png"
    fig.savefig(out, dpi=150); print("wrote", out)


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
    fig.savefig(out, dpi=150); print("wrote", out)


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
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ticks = d["label_positions"]; labels = [str(l) for l in d["labels"]]
    for i, T in enumerate(Ts):
        if f"T{T}_freq" not in d:
            continue
        dist = d[f"T{T}_dist"]; freq = d[f"T{T}_freq"] * 33.356  # THz -> cm^-1
        ax.plot(dist, freq, color=cmap(0.30 + 0.62 * i / (len(Ts) - 1)), lw=1.2, alpha=0.8,
                label=f"T$_{{lat}}$={T} K")
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    for t in ticks[1:-1]:
        ax.axvline(t, color="#ccc", lw=0.6)
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.set_ylabel("frequency [cm$^{-1}$]"); ax.set_ylim(-15, 360)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.set_title("(L) temperature spectrum — 1T-VSe$_2$  (TDEP effective fc$_2$(T), real freqs)", fontsize=10.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = SK / "spectra_L_temperature.png"
    fig.savefig(out, dpi=150); print("wrote", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "E"
    if which in ("E", "both"): plot_E()
    if which in ("L", "both"): plot_L()
    if which in ("Ltdep", "all"): plot_L_tdep()
