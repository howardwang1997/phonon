#!/usr/bin/env python3
"""Build leakage-free P0 graphene validation figures from deployment NPZ files."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def conventional_path(data, key: str):
    """Reorder stored M-G-K-M bands into the conventional G-M-K-G path."""
    bands = np.asarray(data[key])
    dist = np.asarray(data["distances"])
    label_pos = np.asarray(data["label_positions"])
    joins = [int(np.argmin(np.abs(dist - value))) for value in label_pos]
    i_m0, i_g, i_k, i_m1 = joins
    # Reverse M->G, reverse K->M, then reverse G->K.  Drop duplicated joins.
    pieces = [
        bands[..., i_g : i_m0 - 1 if i_m0 else None : -1, :],
        bands[..., i_m1 - 1 : i_k - 1 : -1, :],
        bands[..., i_k - 1 : i_g - 1 : -1, :],
    ]
    out = np.concatenate(pieces, axis=-2)
    lengths = [
        label_pos[1] - label_pos[0],
        label_pos[3] - label_pos[2],
        label_pos[2] - label_pos[1],
    ]
    tick = np.concatenate([[0.0], np.cumsum(lengths)])
    counts = [piece.shape[-2] for piece in pieces]
    xparts = []
    for index, (npoint, lo, hi) in enumerate(zip(counts, tick[:-1], tick[1:])):
        endpoint = index == len(counts) - 1
        xparts.append(np.linspace(lo, hi, npoint, endpoint=endpoint))
    return np.concatenate(xparts), out, tick


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_main(data, outdir: Path):
    x, dft_bands, tick = conventional_path(data, "dft_bands_cm")
    _, method_bands, _ = conventional_path(data, "method_bands_cm")
    dgs = np.asarray(data["dgs_Ry"], float)
    tel = np.asarray(data["T_el_K"], float) / 1000.0
    split = np.asarray(data["split"]).astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2))
    for ax, dg in zip(axes[0], (0.005, 0.02)):
        idx = int(np.argmin(np.abs(dgs - dg)))
        for branch in range(dft_bands.shape[-1] - 2, dft_bands.shape[-1]):
            label_dft = "DFT" if branch == dft_bands.shape[-1] - 1 else None
            label_method = "MLIP + long range" if branch == dft_bands.shape[-1] - 1 else None
            ax.plot(x, dft_bands[idx, :, branch], color="#222222", lw=1.45,
                    label=label_dft, zorder=3)
            ax.plot(x, method_bands[idx, :, branch], color="#D55E00", lw=1.35,
                    ls="--", label=label_method, zorder=2)
        for xpos in tick:
            ax.axvline(xpos, color="#d9d9d9", lw=0.7, zorder=0)
        ax.set_xticks(tick, [r"$\Gamma$", "M", "K", r"$\Gamma$"])
        ax.set_xlim(tick[0], tick[-1])
        ax.set_ylim(1220, 1655)
        tag = "anchor" if split[idx] == "anchor" else "development evidence"
        ax.set_title(f"dg={dg:g} Ry ({tag})", fontsize=10)
        ax.set_ylabel(r"frequency (cm$^{-1}$)")
        style_axis(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[1, 0]
    dft_k = np.asarray(data["dft_kink_K_THz_per_q"], float)
    method_k = np.asarray(data["method_kink_K_THz_per_q"], float)
    anchor = split == "anchor"
    heldout = ~anchor
    ax.plot(tel, dft_k, color="#222222", lw=1.35, marker="o", ms=5,
            label="DFT (8x8)")
    ax.plot(tel, method_k, color="#D55E00", lw=1.35, ls="--", zorder=2,
            label="MLIP + long range")
    ax.scatter(tel[anchor], method_k[anchor], marker="s", s=40, color="#D55E00",
               zorder=4, label="fit anchor")
    ax.scatter(tel[heldout], method_k[heldout], marker="s", s=44,
               facecolor="white", edgecolor="#D55E00", lw=1.4, zorder=5,
               label="development evidence")
    ax.axhline(float(data["backbone_bands_cm"].shape[0]) * 0 +
               float(np.asarray(data["method_kink_K_THz_per_q"])[-1]),
               color="#999999", ls=":", lw=1.0, label="melted limit")
    ax.set_xlabel(r"electronic temperature $T_{el}$ ($10^3$ K)")
    ax.set_ylabel(r"K slope jump $|\Delta(d\omega/dq)|$" + "\n(THz per path-distance)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    style_axis(ax)

    ax = axes[1, 1]
    dft_kito = np.asarray(data["dft_K_iTO_cm"], float)
    method_kito = np.asarray(data["method_K_iTO_cm"], float)
    dft_gamma = np.asarray(data["dft_G_E2g_cm"], float)
    method_gamma = np.asarray(data["method_G_E2g_cm"], float)
    ax.plot(tel, dft_kito, "-o", color="#0072B2", ms=4.5, lw=1.3,
            label=r"DFT K-$i$TO")
    ax.plot(tel, method_kito, "--s", color="#0072B2", mfc="white", ms=4.5,
            lw=1.2, label=r"method K-$i$TO")
    ax.plot(tel, dft_gamma, "-o", color="#7A3E9D", ms=4.5, lw=1.3,
            label=r"DFT $\Gamma$-$E_{2g}$")
    ax.plot(tel, method_gamma, "--s", color="#7A3E9D", mfc="white", ms=4.5,
            lw=1.2, label=r"method $\Gamma$-$E_{2g}$")
    heldout_band_mae = float(np.mean(np.asarray(data["full_band_MAE_cm"])[heldout]))
    heldout_kito_mae = float(np.mean(np.abs(method_kito[heldout] - dft_kito[heldout])))
    heldout_gamma_mae = float(np.mean(np.abs(method_gamma[heldout] - dft_gamma[heldout])))
    ax.text(0.03, 0.48,
            "development-set MAE\n"
            f"full band: {heldout_band_mae:.2f} cm$^{{-1}}$\n"
            f"K-$i$TO: {heldout_kito_mae:.1f} cm$^{{-1}}$\n"
            f"$\\Gamma$-$E_{{2g}}$: {heldout_gamma_mae:.1f} cm$^{{-1}}$",
            transform=ax.transAxes, fontsize=7.5, va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85,
                  "pad": 2.5})
    ax.set_xlabel(r"electronic temperature $T_{el}$ ($10^3$ K)")
    ax.set_ylabel(r"mode frequency (cm$^{-1}$)")
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="center right")
    style_axis(ax)

    fig.suptitle(
        "Graphene P0: 8x8 finite-displacement DFT vs leakage-free MLIP + long-range deployment",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"graphene_8x8_main_p0.{suffix}", dpi=200)
    plt.close(fig)


def plot_diagnostic(data, outdir: Path):
    dgs = np.asarray(data["dgs_Ry"], float)
    split = np.asarray(data["split"]).astype(str)
    anchor = split == "anchor"
    heldout = ~anchor
    dft_k = np.asarray(data["dft_kink_K_THz_per_q"], float)
    method_k = np.asarray(data["method_kink_K_THz_per_q"], float)
    band_mae = np.asarray(data["full_band_MAE_cm"], float)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    ax = axes[0]
    ax.plot(dgs, dft_k, "-o", color="#222222", ms=4, lw=1.25, label="DFT (6x6)")
    ax.plot(dgs, method_k, "--", color="#D55E00", lw=1.3,
            label="MLIP + long range")
    ax.scatter(dgs[anchor], method_k[anchor], marker="s", s=34, color="#D55E00",
               zorder=4, label="fit anchor")
    ax.scatter(dgs[heldout], method_k[heldout], marker="s", s=36, facecolor="white",
               edgecolor="#D55E00", lw=1.2, zorder=5, label="diagnostic point")
    ax.axvspan(0.08, dgs.max(), color="#999999", alpha=0.12,
               label="above background anchor")
    ax.set_xlabel("Fermi-Dirac degauss (Ry)")
    ax.set_ylabel(r"K slope jump (THz per path-distance)")
    ax.legend(frameon=False, fontsize=7.3)
    style_axis(ax)

    ax = axes[1]
    colors = np.where(dgs > 0.08, "#999999", "#0072B2")
    ax.scatter(dgs, band_mae, c=colors, s=32, zorder=3)
    ax.plot(dgs, band_mae, color="#777777", lw=0.8, zorder=1)
    ax.axvline(0.08, color="#777777", ls=":", lw=1.0)
    ax.text(0.084, 0.92 * ax.get_ylim()[1] if ax.get_ylim()[1] else 1,
            "out-of-domain\nextrapolation", color="#666666", fontsize=8, va="top")
    ax.set_xlabel("Fermi-Dirac degauss (Ry)")
    ax.set_ylabel(r"full-band MAE (cm$^{-1}$)")
    style_axis(ax)
    fig.suptitle("Graphene 15-point 6x6 diagnostic: non-anchor points and scope limit",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(outdir / "graphene_15pt_diagnostic_p0.png", dpi=200)
    plt.close(fig)


def plot_prospective(data, outdir: Path):
    """Paper-facing audit: three untouched 8x8 targets plus the frozen curve."""
    x, dft_bands, tick = conventional_path(data, "dft_bands_cm")
    _, method_bands, _ = conventional_path(data, "method_bands_cm")
    dgs = np.asarray(data["dgs_Ry"], float)
    tel = np.asarray(data["T_el_K"], float) / 1000.0
    split = np.asarray(data["split"]).astype(str)
    anchor = split == "anchor"
    prospective = split == "prospective_holdout"
    development = split == "development_holdout"

    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.2))
    ax = axes[0, 0]
    dft_k = np.asarray(data["dft_kink_K_THz_per_q"], float)
    method_k = np.asarray(data["method_kink_K_THz_per_q"], float)
    ax.plot(tel, dft_k, "-o", color="#222222", ms=4.5, lw=1.35,
            label="DFT (8x8)")
    ax.plot(tel, method_k, "--", color="#D55E00", lw=1.4,
            label="frozen MLIP + long range")
    ax.scatter(tel[anchor], method_k[anchor], marker="s", s=42, color="#D55E00",
               zorder=4, label="fit anchor")
    ax.scatter(tel[development], method_k[development], marker="o", s=42,
               facecolor="white", edgecolor="#777777", lw=1.2, zorder=5,
               label="development evidence")
    ax.scatter(tel[prospective], method_k[prospective], marker="D", s=48,
               facecolor="white", edgecolor="#D55E00", lw=1.5, zorder=6,
               label="prospective holdout")
    kink_mae = float(np.mean(np.abs(method_k[prospective] - dft_k[prospective])))
    band_mae = float(np.mean(np.asarray(data["full_band_MAE_cm"])[prospective]))
    ax.text(0.97, 0.96,
            f"prospective MAE\nkink: {kink_mae:.2f} THz/q\nfull band: {band_mae:.2f} cm$^{{-1}}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86,
                  "pad": 2.5})
    ax.set_xlabel(r"electronic temperature $T_{el}$ ($10^3$ K)")
    ax.set_ylabel(r"K slope jump (THz per path-distance)")
    ax.legend(frameon=False, fontsize=7.1, loc="lower left")
    style_axis(ax)

    for ax, dg in zip((axes[0, 1], axes[1, 0], axes[1, 1]), (0.015, 0.03, 0.06)):
        idx = int(np.argmin(np.abs(dgs - dg)))
        for branch in range(dft_bands.shape[-1] - 2, dft_bands.shape[-1]):
            label_dft = "DFT" if branch == dft_bands.shape[-1] - 1 else None
            label_method = "frozen method" if branch == dft_bands.shape[-1] - 1 else None
            ax.plot(x, dft_bands[idx, :, branch], color="#222222", lw=1.4,
                    label=label_dft, zorder=3)
            ax.plot(x, method_bands[idx, :, branch], color="#D55E00", lw=1.3,
                    ls="--", label=label_method, zorder=2)
        for xpos in tick:
            ax.axvline(xpos, color="#d9d9d9", lw=0.7, zorder=0)
        ax.set_xticks(tick, [r"$\Gamma$", "M", "K", r"$\Gamma$"])
        ax.set_xlim(tick[0], tick[-1])
        ax.set_ylim(1220, 1655)
        ax.set_title(f"prospective holdout: dg={dg:g} Ry", fontsize=9.5)
        ax.set_ylabel(r"frequency (cm$^{-1}$)")
        style_axis(ax)
    axes[0, 1].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle(
        "Graphene prospective validation: frozen three-anchor law on unseen 8x8 DFT",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"graphene_8x8_prospective.{suffix}", dpi=200)
    plt.close(fig)


def plot_prospective_full_spectrum(data, outdir: Path):
    """Show all six branches for the three genuinely prospective 8x8 targets."""
    x, dft_bands, tick = conventional_path(data, "dft_bands_cm")
    _, method_bands, _ = conventional_path(data, "method_bands_cm")
    dgs = np.asarray(data["dgs_Ry"], float)
    band_mae = np.asarray(data["full_band_MAE_cm"], float)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    for ax, dg in zip(axes, (0.015, 0.03, 0.06)):
        idx = int(np.argmin(np.abs(dgs - dg)))
        for branch in range(dft_bands.shape[-1]):
            label_dft = "DFT (8x8)" if branch == dft_bands.shape[-1] - 1 else None
            label_method = (
                "frozen method (log law)"
                if branch == dft_bands.shape[-1] - 1
                else None
            )
            ax.plot(
                x,
                dft_bands[idx, :, branch],
                color="#222222",
                lw=1.15,
                alpha=0.88,
                label=label_dft,
                zorder=3,
            )
            ax.plot(
                x,
                method_bands[idx, :, branch],
                color="#D55E00",
                lw=1.05,
                ls="--",
                alpha=0.9,
                label=label_method,
                zorder=2,
            )
        for xpos in tick:
            ax.axvline(xpos, color="#d9d9d9", lw=0.7, zorder=0)
        ax.set_xticks(tick, [r"$\Gamma$", "M", "K", r"$\Gamma$"])
        ax.set_xlim(tick[0], tick[-1])
        ax.set_ylim(-60, 1670)
        ax.set_title(f"prospective dg={dg:g} Ry", fontsize=9.5)
        ax.text(
            0.04,
            0.05,
            f"full-band MAE\n{band_mae[idx]:.2f} cm$^{{-1}}$",
            transform=ax.transAxes,
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82,
                  "pad": 2.0},
        )
        style_axis(ax)
    axes[0].set_ylabel(r"frequency (cm$^{-1}$)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="center left")
    fig.suptitle(
        "Graphene prospective validation: complete six-branch phonon spectrum",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"graphene_8x8_prospective_full_spectrum.{suffix}", dpi=200)
    plt.close(fig)


def plot_law_comparison(log_data, unified_data, outdir: Path):
    """Compare two laws frozen before the prospective labels were inspected."""
    dgs = np.asarray(log_data["dgs_Ry"], float)
    tel = np.asarray(log_data["T_el_K"], float) / 1000.0
    split = np.asarray(log_data["split"]).astype(str)
    prospective = split == "prospective_holdout"
    anchor = split == "anchor"

    unified_dgs = np.asarray(unified_data["dgs_Ry"], float)
    unified_split = np.asarray(unified_data["split"]).astype(str)
    if not np.allclose(dgs, unified_dgs) or not np.array_equal(split, unified_split):
        raise ValueError("comparison inputs do not contain the same evaluation grid")

    dft_k = np.asarray(log_data["dft_kink_K_THz_per_q"], float)
    log_k = np.asarray(log_data["method_kink_K_THz_per_q"], float)
    unified_k = np.asarray(unified_data["method_kink_K_THz_per_q"], float)
    if not np.allclose(
        dft_k,
        np.asarray(unified_data["dft_kink_K_THz_per_q"], float),
        atol=1e-8,
    ):
        raise ValueError("comparison inputs do not share identical DFT targets")

    fig, axes = plt.subplots(1, 3, figsize=(12.7, 3.9))

    ax = axes[0]
    ax.plot(tel, dft_k, "-o", color="#222222", lw=1.45, ms=4.5, label="DFT (8x8)")
    ax.plot(tel, log_k, "--", color="#D55E00", lw=1.5,
            label="log interpolation")
    ax.plot(tel, unified_k, "-.", color="#0072B2", lw=1.5,
            label=r"unified law ($p=4.44,q=3$)")
    ax.scatter(tel[anchor], dft_k[anchor], marker="s", s=38, color="#222222",
               zorder=5, label="shared fit anchors")
    ax.scatter(tel[prospective], dft_k[prospective], marker="D", s=45,
               facecolor="white", edgecolor="#222222", lw=1.25, zorder=6,
               label="prospective DFT")
    ax.set_xlabel(r"electronic temperature $T_{el}$ ($10^3$ K)")
    ax.set_ylabel(r"K slope jump (THz per path-distance)")
    ax.set_title("Kohn-anomaly melting curve", fontsize=10)
    ax.legend(frameon=False, fontsize=7.1, loc="lower left")
    style_axis(ax)

    ax = axes[1]
    xbar = np.arange(int(np.count_nonzero(prospective)))
    width = 0.36
    log_error = np.abs(log_k[prospective] - dft_k[prospective])
    unified_error = np.abs(unified_k[prospective] - dft_k[prospective])
    bars_log = ax.bar(xbar - width / 2, log_error, width, color="#D55E00",
                      label="log interpolation")
    bars_unified = ax.bar(xbar + width / 2, unified_error, width, color="#0072B2",
                          label="unified law")
    for bars in (bars_log, bars_unified):
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(xbar, [f"dg={dg:g}" for dg in dgs[prospective]])
    ax.set_ylabel(r"absolute K-jump error (THz/q)")
    ax.set_title("Prospective anomaly error", fontsize=10)
    ax.legend(frameon=False, fontsize=7.4)
    style_axis(ax)

    def prospective_cm_metrics(data):
        dft_kito = np.asarray(data["dft_K_iTO_cm"], float)
        method_kito = np.asarray(data["method_K_iTO_cm"], float)
        dft_gamma = np.asarray(data["dft_G_E2g_cm"], float)
        method_gamma = np.asarray(data["method_G_E2g_cm"], float)
        return np.array([
            np.mean(np.asarray(data["full_band_MAE_cm"], float)[prospective]),
            np.mean(np.abs(method_kito[prospective] - dft_kito[prospective])),
            np.mean(np.abs(method_gamma[prospective] - dft_gamma[prospective])),
        ])

    ax = axes[2]
    labels = ["full band", r"K-$i$TO", r"$\Gamma$-$E_{2g}$"]
    xbar = np.arange(len(labels))
    log_metrics = prospective_cm_metrics(log_data)
    unified_metrics = prospective_cm_metrics(unified_data)
    bars_log = ax.bar(xbar - width / 2, log_metrics, width, color="#D55E00",
                      label="log interpolation")
    bars_unified = ax.bar(xbar + width / 2, unified_metrics, width, color="#0072B2",
                          label="unified law")
    for bars in (bars_log, bars_unified):
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(xbar, labels)
    ax.set_ylabel(r"prospective MAE (cm$^{-1}$)")
    ax.set_title("Prospective spectral accuracy", fontsize=10)
    ax.legend(frameon=False, fontsize=7.4)
    style_axis(ax)

    fig.suptitle(
        "Frozen-law comparison on untouched graphene targets: anomaly vs spectrum tradeoff",
        fontsize=11.5,
    )
    fig.text(
        0.5,
        0.012,
        "Both candidates use the same three anchors; no prospective target enters either law.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"graphene_8x8_law_comparison.{suffix}", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=Path, default=ROOT / "results" / "p0_graphene")
    parser.add_argument(
        "--prospective-dir",
        type=Path,
        default=ROOT / "results" / "p0_graphene_prospective",
    )
    parser.add_argument(
        "--unified-dir",
        type=Path,
        default=ROOT / "results" / "p0_graphene_prospective_unified",
    )
    args = parser.parse_args()
    args.indir.mkdir(parents=True, exist_ok=True)
    with np.load(args.indir / "graphene_8x8_holdout.npz") as data:
        plot_main(data, args.indir)
    with np.load(args.indir / "graphene_15pt_holdout.npz") as data:
        plot_diagnostic(data, args.indir)
    prospective_npz = args.prospective_dir / "graphene_8x8_holdout.npz"
    if prospective_npz.is_file():
        args.prospective_dir.mkdir(parents=True, exist_ok=True)
        with np.load(prospective_npz) as data:
            plot_prospective(data, args.prospective_dir)
        with np.load(prospective_npz) as data:
            plot_prospective_full_spectrum(data, args.prospective_dir)
    unified_npz = args.unified_dir / "graphene_8x8_holdout.npz"
    if prospective_npz.is_file() and unified_npz.is_file():
        with np.load(prospective_npz) as log_data, np.load(unified_npz) as unified_data:
            plot_law_comparison(log_data, unified_data, args.prospective_dir)
    print(f"wrote P0 figures under {args.indir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
