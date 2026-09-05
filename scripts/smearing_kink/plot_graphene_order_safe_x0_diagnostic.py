#!/usr/bin/env python3
"""Plot the old and atom-order-corrected 450 K q6-SSCHA A' branch near K."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(result_path: Path, acceptance_path: Path) -> dict:
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed" or acceptance.get("converged") is not True:
        raise ValueError(f"SSCHA case did not pass: {acceptance_path}")
    if acceptance["result_sha256"] != sha256(result_path):
        raise ValueError(f"result hash differs from acceptance: {result_path}")
    with np.load(result_path, allow_pickle=False) as data:
        distance = np.asarray(data["distance"], float)
        label_positions = np.asarray(data["label_positions"], float)
        frequency = np.asarray(data["frequency_cm_1"], float)
        lattice_temperature = int(np.asarray(data["lattice_temperature_K"]).reshape(()))
        operator_temperature = int(np.asarray(data["operator_temperature_K"]).reshape(()))
    center = int(np.argmin(np.abs(distance - label_positions[2])))
    return {
        "result": result_path,
        "acceptance": acceptance_path,
        "distance": distance,
        "signed_distance": distance - distance[center],
        "top_frequency": np.max(frequency, axis=1),
        "K_index": center,
        "lattice_temperature_K": lattice_temperature,
        "operator_temperature_K": operator_temperature,
        "degauss_Ry": float(acceptance["condition"]["operator_degauss_Ry"]),
        "result_sha256": sha256(result_path),
        "acceptance_sha256": sha256(acceptance_path),
    }


def rise_at(case: dict, signed_coordinate: float) -> float:
    x = case["signed_distance"]
    y = case["top_frequency"]
    center = case["K_index"]
    if signed_coordinate < 0.0:
        side_x = x[: center + 1]
        side_y = y[: center + 1]
    else:
        side_x = x[center:]
        side_y = y[center:]
    value = float(np.interp(signed_coordinate, side_x, side_y))
    return value - float(y[center])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-result", type=Path, required=True)
    parser.add_argument("--old-acceptance", type=Path, required=True)
    parser.add_argument("--corrected-low-result", type=Path, required=True)
    parser.add_argument("--corrected-low-acceptance", type=Path, required=True)
    parser.add_argument("--corrected-high-result", type=Path, required=True)
    parser.add_argument("--corrected-high-acceptance", type=Path, required=True)
    parser.add_argument("--span", type=float, default=0.12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    old = load_case(args.old_result, args.old_acceptance)
    low = load_case(args.corrected_low_result, args.corrected_low_acceptance)
    high = load_case(args.corrected_high_result, args.corrected_high_acceptance)
    for case in (old, low, high):
        if case["lattice_temperature_K"] != 450:
            raise ValueError("all curves must use lattice temperature 450 K")
    if old["operator_temperature_K"] != low["operator_temperature_K"]:
        raise ValueError("old and corrected low-smearing cases differ in smearing")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"low": "#2666B0", "high": "#D55E00", "old": "#777777"}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    curves = [
        (
            old,
            colors["old"],
            "--",
            f"old invalid interface, degauss={old['degauss_Ry']:.6f} Ry",
            1.5,
        ),
        (
            low,
            colors["low"],
            "-",
            f"order-safe, degauss={low['degauss_Ry']:.6f} Ry",
            2.1,
        ),
        (
            high,
            colors["high"],
            "-",
            f"order-safe, degauss={high['degauss_Ry']:.6f} Ry",
            2.1,
        ),
    ]
    for case, color, style, label, width in curves:
        mask = np.abs(case["signed_distance"]) <= args.span + 1.0e-12
        x = case["signed_distance"][mask]
        y = case["top_frequency"][mask]
        axes[0].plot(x, y, color=color, ls=style, lw=width, label=label)
        axes[1].plot(
            x,
            y - case["top_frequency"][case["K_index"]],
            color=color,
            ls=style,
            lw=width,
        )
    for axis in axes:
        axis.axvline(0.0, color="black", lw=0.7, alpha=0.55)
        axis.grid(alpha=0.18, lw=0.5)
        axis.set_xlabel("signed path distance from K (crystal-coordinate norm)")
        axis.set_xlim(-args.span, args.span)
    axes[0].set_ylabel("A' top-branch frequency (cm$^{-1}$)")
    axes[1].set_ylabel("K-referenced frequency rise (cm$^{-1}$)")
    axes[0].set_title("Absolute q6-SSCHA diagnostic")
    axes[1].set_title("Local shape around K")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(
        "Graphene, lattice temperature 450 K — atom-order correction",
        y=1.17,
        fontsize=11.0,
    )
    fig.text(
        0.5,
        -0.015,
        "q6 sampling-Hessian diagnostic only; full-EPC long-range replacement is not applied here",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "graphene_order_safe_X0_near_K.png"
    pdf = args.output_dir / "graphene_order_safe_X0_near_K.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    cases = {"old_invalid_low_smearing": old, "corrected_low_smearing": low, "corrected_high_smearing": high}
    metrics = {
        "status": "order_safe_X0_diagnostic_plotted",
        "scope": "q6 SSCHA top A-prime branch at fixed lattice temperature 450 K",
        "cases": {
            name: {
                "lattice_temperature_K": case["lattice_temperature_K"],
                "operator_temperature_K": case["operator_temperature_K"],
                "degauss_Ry": case["degauss_Ry"],
                "K_frequency_cm_1": float(case["top_frequency"][case["K_index"]]),
                "left_rise_at_0.05_cm_1": rise_at(case, -0.05),
                "right_rise_at_0.05_cm_1": rise_at(case, 0.05),
                "result": str(case["result"]),
                "result_sha256": case["result_sha256"],
                "acceptance": str(case["acceptance"]),
                "acceptance_sha256": case["acceptance_sha256"],
            }
            for name, case in cases.items()
        },
        "outputs": {
            "png": {"path": str(png), "sha256": sha256(png)},
            "pdf": {"path": str(pdf), "sha256": sha256(pdf)},
        },
    }
    metrics_path = args.output_dir / "graphene_order_safe_X0_near_K.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
