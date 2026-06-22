"""Shared publication style for the composite paper figures (Fig 2/3/4).

Import and call ``set_style()`` at the top of each figure script, then use
``panel(ax, "a")`` for bold panel labels and the ``C`` color dict for a
consistent palette across all figures.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- consistent palette (used across every figure) ----
C = {
    "dfpt": "#000000",        # reference (DFPT / experiment)
    "mace": "#d1495b",        # MACE baseline (the model we fine-tune)
    "baseline": "#d1495b",    # baseline MLIP (== MACE)
    "finetuned": "#2a9d4a",   # FC-distilled / fine-tuned
    "mattersim": "#1f77b4",
    "sevennet": "#e58f1a",
    "coverage": "#2a9d4a",    # coverage acquisition (best)
    "random": "#7f7f7f",
    "uncertainty": "#d1495b",  # naive uncertainty (worst)
    "breadth": "#6a3d9a",
    "depth": "#2a9d4a",
}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    })


def panel(ax, letter: str, dx: float = -0.085, dy: float = 1.04) -> None:
    """Bold lower-case panel label (a, b, c ...) at the top-left of an axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="right")
