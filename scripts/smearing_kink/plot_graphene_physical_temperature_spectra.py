"""Plot the formal graphene L0/Q0 phonon spectra at 300, 450, and 600 K.

The figure uses only the accepted diagonal physical-temperature calculations:

* L0: pooled classical MD/TDEP spectrum with the matched long-range operator.
* Q0: converged quantum SSCHA free-energy-Hessian spectrum.

Run with:

    conda run -n phonon python \
        scripts/smearing_kink/plot_graphene_physical_temperature_spectra.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
S0 = (
    ROOT
    / "results"
    / "graphene_physics_temperature"
    / "post_p4_feasibility"
    / "S0_unified_short"
)
OUT = S0 / "figures"
PDF_OUT = ROOT / "output" / "pdf"
TEMPERATURES = (300, 450, 600)
COLORS = {
    300: "#0072B2",
    450: "#E69F00",
    600: "#D55E00",
}


def _scalar(array: np.ndarray) -> float:
    return float(np.asarray(array).reshape(()))


def _load() -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
    l0: dict[int, dict[str, np.ndarray]] = {}
    q0: dict[int, dict[str, np.ndarray]] = {}
    for temperature in TEMPERATURES:
        l0_path = S0 / "L0_classical_tdep" / f"T{temperature}" / "final_physical_tdep.npz"
        q0_path = S0 / "Q0_quantum_sscha" / f"formal_T{temperature}" / "result.npz"
        if not l0_path.is_file() or not q0_path.is_file():
            raise FileNotFoundError(f"Missing formal result at {temperature} K: {l0_path} or {q0_path}")

        with np.load(l0_path, allow_pickle=False) as data:
            l0[temperature] = {key: np.array(data[key]) for key in data.files}
        with np.load(q0_path, allow_pickle=False) as data:
            q0[temperature] = {key: np.array(data[key]) for key in data.files}

        if int(_scalar(l0[temperature]["temperature_K"])) != temperature:
            raise ValueError(f"L0 temperature metadata mismatch at {temperature} K")
        if int(_scalar(q0[temperature]["lattice_temperature_K"])) != temperature:
            raise ValueError(f"Q0 lattice-temperature metadata mismatch at {temperature} K")
        if int(_scalar(q0[temperature]["operator_temperature_K"])) != temperature:
            raise ValueError(f"Q0 operator-temperature metadata mismatch at {temperature} K")
        if not bool(np.asarray(q0[temperature]["converged"]).reshape(())):
            raise ValueError(f"Q0 result is not converged at {temperature} K")

        for channel, distance_key, frequency_key, payload in (
            ("L0", "pooled_dist", "pooled_frequency_cm-1", l0[temperature]),
            ("Q0", "distance", "frequency_cm_1", q0[temperature]),
        ):
            distance = payload[distance_key]
            frequency = payload[frequency_key]
            if distance.ndim != 1 or frequency.shape != (distance.size, 6):
                raise ValueError(f"Unexpected {channel} array shape at {temperature} K")
            if not np.all(np.isfinite(distance)) or not np.all(np.isfinite(frequency)):
                raise ValueError(f"Non-finite {channel} data at {temperature} K")

    return l0, q0


def _format_axis(ax: plt.Axes, payload: dict[str, np.ndarray], *, full: bool) -> None:
    ticks = np.asarray(payload["label_positions"], dtype=float)
    labels = ["M", r"$\Gamma$", "K", "M"]
    ax.set_xlim(float(ticks[0]), float(ticks[-1]))
    ax.set_xticks(ticks, labels)
    for position in ticks[1:-1]:
        ax.axvline(position, color="#B8B8B8", lw=0.7, zorder=0)
    if full:
        ax.axhline(0.0, color="#777777", lw=0.7, zorder=0)
        ax.set_ylim(-50.0, 1650.0)
    else:
        ax.set_ylim(1175.0, 1625.0)
    ax.grid(axis="y", color="#E6E6E6", lw=0.55)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_channel(
    full_ax: plt.Axes,
    top_ax: plt.Axes,
    data: dict[int, dict[str, np.ndarray]],
    *,
    distance_key: str,
    frequency_key: str,
    title: str,
) -> None:
    for temperature in TEMPERATURES:
        payload = data[temperature]
        distance = np.asarray(payload[distance_key], dtype=float)
        frequency = np.asarray(payload[frequency_key], dtype=float)
        color = COLORS[temperature]
        full_ax.plot(distance, frequency, color=color, lw=0.85, alpha=0.78)
        top_ax.plot(distance, frequency[:, -1], color=color, lw=1.8, alpha=0.98)

    _format_axis(full_ax, data[450], full=True)
    _format_axis(top_ax, data[450], full=False)
    full_ax.set_title(title, fontsize=11.5, pad=8)
    top_ax.set_title("Highest optical branch", fontsize=10.5, pad=7)


def _high_symmetry_summary(
    data: dict[int, dict[str, np.ndarray]], distance_key: str, frequency_key: str
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for temperature in TEMPERATURES:
        payload = data[temperature]
        distance = np.asarray(payload[distance_key], dtype=float)
        frequency = np.asarray(payload[frequency_key], dtype=float)
        labels = [str(value) for value in payload["labels"]]
        positions = np.asarray(payload["label_positions"], dtype=float)
        row: dict[str, float] = {}
        for label, position in zip(labels, positions, strict=True):
            clean_label = "Gamma" if "Gamma" in label else label.replace("$", "")
            if clean_label in row:
                clean_label = f"{clean_label}_end"
            index = int(np.argmin(np.abs(distance - position)))
            row[clean_label] = float(frequency[index, -1])
        result[str(temperature)] = row
    return result


def main() -> None:
    l0, q0 = _load()
    OUT.mkdir(parents=True, exist_ok=True)
    PDF_OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.4, 8.0),
        sharex="col",
        gridspec_kw={"height_ratios": (1.65, 1.0)},
    )
    _plot_channel(
        axes[0, 0],
        axes[1, 0],
        l0,
        distance_key="pooled_dist",
        frequency_key="pooled_frequency_cm-1",
        title="(a) Classical MD/TDEP (L0)",
    )
    _plot_channel(
        axes[0, 1],
        axes[1, 1],
        q0,
        distance_key="distance",
        frequency_key="frequency_cm_1",
        title="(b) Quantum SSCHA (Q0)",
    )

    axes[0, 0].set_ylabel(r"Frequency (cm$^{-1}$)")
    axes[1, 0].set_ylabel(r"Frequency (cm$^{-1}$)")
    axes[1, 0].set_xlabel("Wave-vector path")
    axes[1, 1].set_xlabel("Wave-vector path")

    handles = []
    for temperature in TEMPERATURES:
        degauss = _scalar(l0[temperature]["degauss_Ry"])
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[temperature],
                lw=2.2,
                label=rf"{temperature} K / {degauss:.5f} Ry",
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        fontsize=8.7,
        handlelength=2.4,
        columnspacing=1.5,
        title="lattice temperature / smearing/degauss",
        title_fontsize=8.8,
    )
    fig.suptitle(
        "Graphene phonon spectra under diagonal physical-temperature conditions",
        fontsize=13.0,
        y=0.985,
    )
    fig.text(
        0.5,
        0.948,
        "The lattice temperature and the matched electronic smearing operator vary together.",
        ha="center",
        va="center",
        fontsize=9.3,
        color="#444444",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.905, bottom=0.145, hspace=0.23, wspace=0.13)

    png = OUT / "graphene_phonon_spectra_temperature_L0_Q0.png"
    pdf = OUT / "graphene_phonon_spectra_temperature_L0_Q0.pdf"
    delivery_pdf = PDF_OUT / "graphene_phonon_spectra_temperature_L0_Q0.pdf"
    fig.savefig(png, dpi=240, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(delivery_pdf, facecolor="white")
    plt.close(fig)

    summary = {
        "scope": "accepted diagonal physical-temperature results",
        "temperatures_K": list(TEMPERATURES),
        "smearing_degauss_Ry": {
            str(temperature): _scalar(l0[temperature]["degauss_Ry"])
            for temperature in TEMPERATURES
        },
        "L0_highest_optical_cm-1": _high_symmetry_summary(
            l0, "pooled_dist", "pooled_frequency_cm-1"
        ),
        "Q0_highest_optical_cm-1": _high_symmetry_summary(q0, "distance", "frequency_cm_1"),
        "outputs": [
            str(png.relative_to(ROOT)),
            str(pdf.relative_to(ROOT)),
            str(delivery_pdf.relative_to(ROOT)),
        ],
    }
    summary_path = OUT / "graphene_phonon_spectra_temperature_L0_Q0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(png)
    print(pdf)
    print(delivery_pdf)
    print(summary_path)


if __name__ == "__main__":
    main()
