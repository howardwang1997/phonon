#!/usr/bin/env python3
"""Plot the weekly Zero Smearing and finite-lattice graphene summary.

The Zero Smearing panel uses the frozen static-lattice E44 full-EPC curve.
The remaining panels use the force-gated S0 result from the final R2AT
300/450/600 K acceptance at one fixed electronic smearing.  R2AO appears only
as the paired spectral-sensitivity control in the K-frequency panel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
ZERO_DIR = BASE / "E44_epc_zero_wider_curve"
FINITE_DIR = (
    BASE
    / "R2R_multipolar_background"
    / "R2AT_paired_full_epc_acceptance_20260827"
)
DEFAULT_OUTPUT = FINITE_DIR / "graphene_zero_smearing_and_finite_lattice_summary.png"
TEMPERATURES = (300, 450, 600)
TEMPERATURE_COLORS = {
    300: "#2C7BB6",
    450: "#D95F02",
    600: "#7B4AB0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_inputs() -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    zero = pd.read_csv(ZERO_DIR / "zero_dense_curve.csv")
    zero_summary = json.loads(
        (ZERO_DIR / "zero_dense_curve_summary.json").read_text(encoding="utf-8")
    )
    finite = pd.read_csv(FINITE_DIR / "r2ao_s0_fixed_smearing_full_epc.csv")
    finite_summary = json.loads(
        (FINITE_DIR / "r2ao_full_epc_acceptance_summary.json").read_text(
            encoding="utf-8"
        )
    )

    if zero_summary["status"] != "smooth_zero_full_EPC_curve_generated":
        raise ValueError("Zero Smearing result does not have the frozen E44 status")
    expected_status = "R2AT_PAIRED_QUANTITATIVE_KOHN_ANOMALY_SENSITIVITY_PASSED"
    if finite_summary["status"] != expected_status:
        raise ValueError("finite-lattice result did not pass the final R2AT acceptance")
    if finite_summary["primary_force_gated_short_model"] != "S0":
        raise ValueError("the force-gated primary model is no longer S0")
    if tuple(finite_summary["lattice_temperatures_K"]) != TEMPERATURES:
        raise ValueError("finite-lattice temperature set changed")
    if not np.isclose(
        finite_summary["electronic_control"]["smearing_degauss_Ry"],
        0.0019000869,
        atol=1.0e-14,
        rtol=0.0,
    ):
        raise ValueError("finite-lattice electronic smearing changed")

    required_zero = {
        "direction",
        "q1",
        "q2",
        "MLIP_plus_full_EPC_cm-1",
    }
    required_finite = {
        "short_model",
        "lattice_temperature_K",
        "direction",
        "signed_q_2pi_over_a",
        "full_EPC_cm-1",
    }
    if not required_zero.issubset(zero.columns):
        raise ValueError("Zero Smearing CSV schema changed")
    if not required_finite.issubset(finite.columns):
        raise ValueError("finite-lattice CSV schema changed")
    if not np.isfinite(zero["MLIP_plus_full_EPC_cm-1"]).all():
        raise ValueError("Zero Smearing curve contains non-finite values")
    if not np.isfinite(finite["full_EPC_cm-1"]).all():
        raise ValueError("finite-lattice curve contains non-finite values")
    return zero, zero_summary, finite, finite_summary


def add_zero_signed_q(zero: pd.DataFrame) -> pd.DataFrame:
    """Convert fractional reciprocal coordinates to signed |q-K|/(2pi/a)."""
    converted = zero.copy()
    du = converted["q1"].to_numpy(float) - 1.0 / 3.0
    dv = converted["q2"].to_numpy(float) - 1.0 / 3.0
    # Graphene reciprocal basis vectors have a 60-degree included angle and
    # magnitude 2/sqrt(3) in units of 2pi/a.
    magnitude = (2.0 / np.sqrt(3.0)) * np.sqrt(du * du + dv * dv + du * dv)
    sign = np.where(
        converted["direction"].eq("KG"),
        -1.0,
        np.where(converted["direction"].eq("KM"), 1.0, 0.0),
    )
    converted["signed_q_2pi_over_a"] = sign * magnitude
    return converted.sort_values("signed_q_2pi_over_a")


def finite_curve(
    finite: pd.DataFrame, model: str, temperature: int
) -> pd.DataFrame:
    selected = finite[
        finite["short_model"].eq(model)
        & finite["lattice_temperature_K"].eq(temperature)
    ].copy()
    selected = selected.sort_values("signed_q_2pi_over_a")
    if len(selected) != 241:
        raise ValueError(f"{model} {temperature} K curve does not contain 241 points")
    return selected


def k_frequency(curve: pd.DataFrame) -> float:
    index = np.abs(curve["signed_q_2pi_over_a"].to_numpy(float)).argmin()
    x = float(curve.iloc[index]["signed_q_2pi_over_a"])
    if abs(x) > 1.0e-12:
        raise ValueError("finite-lattice curve does not contain K exactly")
    return float(curve.iloc[index]["full_EPC_cm-1"])


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.65)
    axis.axvline(0.0, color="#A0A0A0", linewidth=0.9, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", width=0.8)


def make_plot(
    zero: pd.DataFrame,
    zero_summary: dict,
    finite: pd.DataFrame,
    finite_summary: dict,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.7,
            "figure.dpi": 150,
            "savefig.dpi": 240,
        }
    )
    zero = add_zero_signed_q(zero)
    zero_k = float(zero_summary["K_anchor_frequency_cm-1"])
    finite_curves = {
        (model, temperature): finite_curve(finite, model, temperature)
        for model in ("S0", "R2AO")
        for temperature in TEMPERATURES
    }
    finite_k = {
        key: k_frequency(curve) for key, curve in finite_curves.items()
    }

    summary_metrics = finite_summary[
        "metrics_by_short_model_and_lattice_temperature_K"
    ]
    for temperature in TEMPERATURES:
        expected = float(
            summary_metrics["S0"][str(temperature)]["full_epc"][
                "K_frequency_cm-1"
            ]
        )
        if not np.isclose(finite_k[("S0", temperature)], expected, atol=1.0e-9):
            raise ValueError(f"S0 {temperature} K CSV and summary disagree")

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6))
    ax_zero, ax_absolute, ax_shape, ax_temperature = axes.ravel()

    # (a) Frozen static-lattice Zero Smearing result.
    zero_colors = {"KG": "#207567", "KM": "#7047A8"}
    for direction, label in (("KG", "K→Γ"), ("KM", "K→M")):
        curve = zero[zero["direction"].eq(direction)]
        curve = curve[np.abs(curve["signed_q_2pi_over_a"]) <= 0.0300001]
        ax_zero.plot(
            curve["signed_q_2pi_over_a"],
            curve["MLIP_plus_full_EPC_cm-1"] - zero_k,
            color=zero_colors[direction],
            linewidth=2.5,
            label=f"full-EPC dense curve ({label})",
        )
    ax_zero.scatter([0.0], [0.0], s=30, color="black", zorder=6, label="K anchor")

    d003 = zero_summary["d003_depths"]
    extrapolated_depth = np.asarray(
        [
            d003["KG"]["frozen_resolution_extrapolated_depth_cm-1"],
            d003["KM"]["frozen_resolution_extrapolated_depth_cm-1"],
        ]
    )
    extrapolated_uncertainty = np.asarray(
        [
            d003["KG"]["frozen_numerical_uncertainty_cm-1"],
            d003["KM"]["frozen_numerical_uncertainty_cm-1"],
        ]
    )
    dfpt_depth = np.asarray(
        [
            d003["KG"]["DFPT_1_over_k_limit_depth_cm-1"],
            d003["KM"]["DFPT_1_over_k_limit_depth_cm-1"],
        ]
    )
    q_d003 = np.asarray([-0.002, 0.002])
    ax_zero.errorbar(
        q_d003,
        extrapolated_depth,
        yerr=extrapolated_uncertainty,
        fmt="o",
        markersize=5.5,
        color="#D95F02",
        capsize=3,
        linewidth=1.2,
        zorder=7,
        label="full-EPC resolution extrapolation",
    )
    ax_zero.scatter(
        q_d003,
        dfpt_depth,
        marker="s",
        s=39,
        facecolor="none",
        edgecolor="black",
        linewidth=1.1,
        zorder=8,
        label=r"DFPT $1/N_k$ extrapolation",
    )
    ax_zero.set_xlim(-0.031, 0.031)
    ax_zero.set_ylim(-0.7, None)
    ax_zero.set_title("(a) Zero Smearing: static-lattice anomaly")
    ax_zero.set_xlabel(r"signed $|q-K|/(2\pi/a)$")
    ax_zero.set_ylabel(r"$\omega(q)-\omega(K)$ (cm$^{-1}$)")
    ax_zero.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=1,
        labelspacing=0.35,
    )
    style_axis(ax_zero)

    # (b) Force-gated finite-lattice S0 absolute spectra.
    for temperature in TEMPERATURES:
        curve = finite_curves[("S0", temperature)]
        ax_absolute.plot(
            curve["signed_q_2pi_over_a"],
            curve["full_EPC_cm-1"],
            color=TEMPERATURE_COLORS[temperature],
            linewidth=2.5,
            label=f"{temperature} K",
        )
        ax_absolute.scatter(
            [0.0],
            [finite_k[("S0", temperature)]],
            color=TEMPERATURE_COLORS[temperature],
            s=27,
            zorder=5,
        )
    ax_absolute.set_xlim(-0.041, 0.041)
    ax_absolute.set_title("(b) Finite lattice: force-gated S0")
    ax_absolute.set_xlabel(r"signed $|q-K|/(2\pi/a)$")
    ax_absolute.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    ax_absolute.legend(loc="upper center", frameon=False, ncol=3)
    style_axis(ax_absolute)

    # (c) The same S0 curves referenced to their own K frequencies.
    for temperature in TEMPERATURES:
        curve = finite_curves[("S0", temperature)]
        window = np.abs(curve["signed_q_2pi_over_a"].to_numpy(float)) <= 0.0250001
        ax_shape.plot(
            curve.loc[window, "signed_q_2pi_over_a"],
            curve.loc[window, "full_EPC_cm-1"] - finite_k[("S0", temperature)],
            color=TEMPERATURE_COLORS[temperature],
            linewidth=2.5,
            label=f"{temperature} K",
        )
    static_validation = finite_summary["static_full_EPC_DFPT_validation"]["metrics"]
    ax_shape.text(
        0.5,
        0.95,
        "static finite-smearing DFPT gate\n"
        f"shape RMSE / max = {static_validation['K_referenced_shape_RMSE_cm-1']:.3f} / "
        f"{static_validation['K_referenced_shape_max_abs_cm-1']:.3f} cm$^{{-1}}$",
        transform=ax_shape.transAxes,
        ha="center",
        va="top",
        fontsize=8.8,
        color="#3A3A3A",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )
    ax_shape.set_xlim(-0.026, 0.026)
    ax_shape.set_ylim(-0.7, None)
    ax_shape.set_title("(c) Finite lattice: K-referenced shape")
    ax_shape.set_xlabel(r"signed $|q-K|/(2\pi/a)$")
    ax_shape.set_ylabel(r"$\omega(q)-\omega(K)$ (cm$^{-1}$)")
    style_axis(ax_shape)

    # (d) S0 is the primary curve; R2AO is a paired spectral sensitivity only.
    temperature_values = np.asarray(TEMPERATURES, float)
    s0_values = np.asarray([finite_k[("S0", value)] for value in TEMPERATURES])
    r2ao_values = np.asarray([finite_k[("R2AO", value)] for value in TEMPERATURES])
    ax_temperature.plot(
        temperature_values,
        s0_values,
        color="#202020",
        marker="s",
        markersize=6.2,
        linewidth=2.4,
        label="S0 (force-gated primary)",
    )
    ax_temperature.plot(
        temperature_values,
        r2ao_values,
        color="#777777",
        marker="o",
        markersize=5.8,
        linewidth=2.0,
        linestyle="--",
        label="R2AO (paired sensitivity)",
    )
    for left, right in ((0, 1), (1, 2)):
        midpoint_x = 0.5 * (temperature_values[left] + temperature_values[right])
        midpoint_y = 0.5 * (s0_values[left] + s0_values[right]) + 0.8
        shift = s0_values[right] - s0_values[left]
        ax_temperature.text(
            midpoint_x,
            midpoint_y,
            f"S0 Δ = {shift:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="#202020",
        )
    max_model_difference = float(np.max(np.abs(r2ao_values - s0_values)))
    ax_temperature.text(
        0.04,
        0.05,
        f"max |R2AO−S0| = {max_model_difference:.2f} cm$^{{-1}}$",
        transform=ax_temperature.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#444444",
    )
    ax_temperature.set_xlim(280, 620)
    ax_temperature.set_title("(d) K frequency and paired sensitivity")
    ax_temperature.set_xlabel("lattice temperature (K)")
    ax_temperature.set_ylabel(r"K frequency (cm$^{-1}$)")
    ax_temperature.legend(loc="upper right", frameon=False)
    ax_temperature.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.65)
    ax_temperature.spines["top"].set_visible(False)
    ax_temperature.spines["right"].set_visible(False)

    fig.suptitle(
        "Graphene K–A$'$ Kohn anomaly: Zero Smearing and finite lattice temperature\n"
        "static lattice: smearing/degauss = 0 Ry  |  finite lattice: "
        "smearing/degauss = 0.0019000869 Ry",
        fontsize=16,
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Zero Smearing d = 0.003 corresponds to |q−K|/(2π/a) = 0.002.  "
        "Finite-lattice curves use 300 SSCHA configurations at each lattice temperature; "
        "electronic smearing is held fixed.",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#4A4A4A",
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.095,
        top=0.865,
        wspace=0.23,
        hspace=0.34,
    )

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


def main() -> None:
    args = parse_args()
    zero, zero_summary, finite, finite_summary = load_inputs()
    make_plot(zero, zero_summary, finite, finite_summary, args.output)


if __name__ == "__main__":
    main()
