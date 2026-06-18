"""Plotting helpers: overlay MLIP vs DFPT phonon dispersions (paper figures).

Both ``PhononResult`` objects must come from the same primitive cell / band
settings so the seekpath path matches.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _segment_label(lbl) -> str:
    s = str(lbl)
    return {"GAMMA": "$\\Gamma$", "G": "$\\Gamma$"}.get(s.upper(), s)


def plot_three_way(
    dfpt,
    baseline,
    finetuned,
    out: str = "results/figures/three_way.png",
    title: Optional[str] = None,
):
    """Overlay DFPT (black dashed) vs baseline (red) vs fine-tuned (blue).

    The A3 before/after figure: shows the fine-tuned curves snapping back onto
    the DFPT reference while the baseline sits softened below.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    def _plot(result, color, ls, label, lw=1.2):
        first = True
        for d, f in zip(result.band_distances, result.band_frequencies):
            d = np.asarray(d)
            f = np.asarray(f)
            for b in range(f.shape[1]):
                ax.plot(d, f[:, b], color=color, ls=ls, lw=lw,
                        label=label if first else None)
                first = False

    _plot(dfpt, "k", "--", "DFPT (reference)")
    _plot(baseline, "tab:red", "-", "MACE-MP-0 (baseline)", lw=1.0)
    _plot(finetuned, "tab:blue", "-", "fine-tuned (FC-distilled)")

    boundaries = [np.asarray(d)[0] for d in dfpt.band_distances]
    boundaries.append(np.asarray(dfpt.band_distances[-1])[-1])
    for x in boundaries:
        ax.axvline(x, color="0.88", lw=0.8, zorder=0)
    ax.axhline(0.0, color="0.6", lw=0.6, ls=":")
    ax.set_ylabel("Frequency (THz)")
    ax.set_xlim(boundaries[0], boundaries[-1])
    ax.set_title(title or f"{dfpt.formula} — phonon dispersion before/after fine-tuning")
    ax.legend(loc="lower center", fontsize=9, ncol=3)
    fig.tight_layout()

    import os

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_dispersion_comparison(
    pred,
    ref=None,
    out: str = "results/dispersion.png",
    title: Optional[str] = None,
    pred_label: str = "MLIP",
    ref_label: str = "DFPT",
):
    """Overlay predicted (solid) and reference (dashed) phonon dispersions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))

    def _plot(result, color, ls, label):
        dists = result.band_distances
        freqs = result.band_frequencies
        first = True
        for d, f in zip(dists, freqs):
            d = np.asarray(d)
            f = np.asarray(f)  # (n_q, n_band)
            for b in range(f.shape[1]):
                ax.plot(d, f[:, b], color=color, ls=ls, lw=1.1,
                        label=label if first else None)
                first = False

    if ref is not None:
        _plot(ref, "0.4", "--", f"{ref_label}")
    _plot(pred, "tab:red", "-", f"{pred_label}")

    # segment boundaries + labels
    boundaries = [np.asarray(d)[0] for d in pred.band_distances]
    boundaries.append(np.asarray(pred.band_distances[-1])[-1])
    for x in boundaries:
        ax.axvline(x, color="0.85", lw=0.8, zorder=0)
    ax.axhline(0.0, color="0.6", lw=0.6, ls=":")

    if pred.band_labels:
        # phonopy labels: flat list aligned to connected segments
        labels = [_segment_label(l) for l in np.ravel(pred.band_labels)]
        ticks = boundaries[: len(labels)]
        ax.set_xticks(ticks[: len(labels)])
        ax.set_xticklabels(labels[: len(ticks)])

    ax.set_ylabel("Frequency (THz)")
    ax.set_xlim(boundaries[0], boundaries[-1])
    ax.set_title(title or f"{pred.formula}  phonon dispersion")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    import os

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
