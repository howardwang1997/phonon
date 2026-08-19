#!/usr/bin/env python3
"""Plot a controlled lattice-temperature comparison at one fixed smearing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results"
    / "graphene_physics_temperature"
    / "post_p4_feasibility"
    / "S0_unified_short"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t300-result",
        type=Path,
        default=BASE / "Q0_quantum_sscha/formal_T300/result.npz",
    )
    parser.add_argument(
        "--t300-acceptance",
        type=Path,
        default=BASE / "Q0_quantum_sscha/formal_T300/acceptance.json",
    )
    parser.add_argument(
        "--t450-result",
        type=Path,
        default=BASE / "X0_cross_development/sscha_Tlat450_Tel300/result.npz",
    )
    parser.add_argument(
        "--t450-acceptance",
        type=Path,
        default=BASE
        / "X0_cross_development/sscha_Tlat450_Tel300/acceptance.json",
    )
    parser.add_argument(
        "--x0-acceptance",
        type=Path,
        default=BASE / "X0_cross_development/analysis/sscha_acceptance.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "graphene_physics_temperature"
        / "post_p4_feasibility"
        / "E47_fixed_smearing_lattice_temperature_comparison",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case(result_path: Path, acceptance_path: Path) -> dict:
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed" or acceptance.get("converged") is not True:
        raise RuntimeError(f"SSCHA result did not pass: {acceptance_path}")
    if acceptance.get("result_sha256") != sha256(result_path):
        raise RuntimeError(f"result hash mismatch: {result_path}")

    with np.load(result_path, allow_pickle=False) as payload:
        lattice_temperature = int(payload["lattice_temperature_K"])
        operator_temperature = int(payload["operator_temperature_K"])
        distance = np.asarray(payload["distance"], float)
        label_positions = np.asarray(payload["label_positions"], float)
        labels = np.asarray(payload["labels"])
        frequency = np.asarray(payload["frequency_cm_1"], float)
        converged = bool(payload["converged"])
        n_configs = int(payload["n_configs"])
        final_population = int(payload["final_population"])
    if not converged or not np.isfinite(frequency).all():
        raise RuntimeError(f"invalid SSCHA spectrum: {result_path}")
    if list(labels) != ["M", "$\\Gamma$", "K", "M"]:
        raise RuntimeError(f"unexpected path labels in {result_path}: {labels}")
    condition = acceptance["condition"]
    if int(condition["lattice_temperature_K"]) != lattice_temperature:
        raise RuntimeError("lattice-temperature metadata mismatch")
    if int(condition["operator_temperature_K"]) != operator_temperature:
        raise RuntimeError("operator-temperature metadata mismatch")

    K_position = float(label_positions[2])
    Gamma_position = float(label_positions[1])
    q_scale = (2.0 / 3.0) / (K_position - Gamma_position)
    signed_q = (distance - K_position) * q_scale
    K_index = int(np.argmin(np.abs(signed_q)))
    if not np.isclose(signed_q[K_index], 0.0, atol=1.0e-12):
        raise RuntimeError("K is absent from the band path")
    top = frequency[:, -1]
    return {
        "result_path": result_path,
        "acceptance_path": acceptance_path,
        "result_sha256": sha256(result_path),
        "acceptance_sha256": sha256(acceptance_path),
        "lattice_temperature_K": lattice_temperature,
        "operator_temperature_K": operator_temperature,
        "smearing_degauss_Ry": float(condition["operator_degauss_Ry"]),
        "signed_q": signed_q,
        "top_cm-1": top,
        "K_index": K_index,
        "K_frequency_cm-1": float(top[K_index]),
        "n_configs": n_configs,
        "final_population": final_population,
        "supercell_min_frequency_cm-1": float(
            acceptance["supercell_min_frequency_cm-1"]
        ),
        "imaginary_modes_below_minus_1_cm-1": int(
            acceptance["supercell_imaginary_modes_below_minus_1_cm-1"]
        ),
    }


def side_rises(case: dict, distances: tuple[float, ...]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for direction, sign in (("KG", -1.0), ("KM", 1.0)):
        signed_q = np.asarray(case["signed_q"], float)
        select = signed_q * sign >= -1.0e-12
        radial = signed_q[select] * sign
        frequency = np.asarray(case["top_cm-1"], float)[select]
        order = np.argsort(radial)
        output[direction] = {
            f"d{distance:.3f}_rise_cm-1": float(
                np.interp(distance, radial[order], frequency[order])
                - case["K_frequency_cm-1"]
            )
            for distance in distances
        }
    return output


def plot(cases: list[dict], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ("#0072B2", "#D55E00")
    figure, (absolute_axis, shape_axis) = plt.subplots(1, 2, figsize=(12.2, 4.65))
    for case, color in zip(cases, colors):
        select = np.abs(case["signed_q"]) <= 0.04 + 1.0e-12
        label = f"{case['lattice_temperature_K']} K lattice temperature"
        absolute_axis.plot(
            case["signed_q"][select],
            case["top_cm-1"][select],
            color=color,
            lw=2.35,
            label=label,
        )
        shape_axis.plot(
            case["signed_q"][select],
            case["top_cm-1"][select] - case["K_frequency_cm-1"],
            color=color,
            lw=2.35,
            label=label,
        )

    for axis in (absolute_axis, shape_axis):
        axis.axvline(0.0, color="#999999", lw=0.85, zorder=0)
        axis.grid(alpha=0.20)
        axis.set_xlim(-0.04, 0.04)
        axis.set_xlabel(r"signed $|q-K|/(2\pi/a)$  (K→Γ < 0; K→M > 0)")
        axis.text(
            0.5,
            0.965,
            r"$\Gamma\ \leftarrow\ K\ \rightarrow\ M$",
            transform=axis.transAxes,
            ha="center",
            va="top",
            color="#555555",
            fontsize=9.3,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    absolute_values = [
        case["top_cm-1"][np.abs(case["signed_q"]) <= 0.04 + 1.0e-12]
        for case in cases
    ]
    absolute_axis.set_ylim(
        min(float(np.min(values)) for values in absolute_values) - 0.5,
        max(float(np.max(values)) for values in absolute_values) + 0.5,
    )
    shape_values = [
        case["top_cm-1"][np.abs(case["signed_q"]) <= 0.04 + 1.0e-12]
        - case["K_frequency_cm-1"]
        for case in cases
    ]
    shape_axis.set_ylim(-0.15, 1.06 * max(float(np.max(values)) for values in shape_values))

    absolute_axis.set_ylabel(r"A$_1'$ frequency (cm$^{-1}$)")
    shape_axis.set_ylabel(r"K-referenced rise, $\omega(q)-\omega(K)$ (cm$^{-1}$)")
    absolute_axis.set_title("(a) Absolute frequency")
    shape_axis.set_title("(b) K-referenced shape")
    handles, labels = absolute_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Graphene K-point A$_1'$ at fixed electronic smearing",
        y=0.985,
        fontsize=13.3,
    )
    figure.text(
        0.5,
        0.025,
        "Quantum SSCHA; fixed cell; smearing/degauss = 0.0019000869 Ry for both curves",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#555555",
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.79, wspace=0.24)
    figure.savefig(output, dpi=240, facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    cases = [
        load_case(args.t300_result, args.t300_acceptance),
        load_case(args.t450_result, args.t450_acceptance),
    ]
    smearings = {case["smearing_degauss_Ry"] for case in cases}
    operator_temperatures = {case["operator_temperature_K"] for case in cases}
    if len(smearings) != 1 or operator_temperatures != {300}:
        raise RuntimeError("comparison does not hold electronic smearing fixed")
    if [case["lattice_temperature_K"] for case in cases] != [300, 450]:
        raise RuntimeError("expected 300 K and 450 K lattice-temperature cases")

    x0_acceptance = json.loads(args.x0_acceptance.read_text(encoding="utf-8"))
    if x0_acceptance.get("status") != "cross_term_exceeds_fixed_gate":
        raise RuntimeError("unexpected X0 separability-gate status")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "fixed_smearing_lattice_temperature_comparison.png"
    plot(cases, png_path)

    sample_distances = (0.003, 0.008, 0.015, 0.025, 0.040)
    case_metrics = []
    for case in cases:
        case_metrics.append(
            {
                "lattice_temperature_K": case["lattice_temperature_K"],
                "operator_temperature_K": case["operator_temperature_K"],
                "smearing_degauss_Ry": case["smearing_degauss_Ry"],
                "K_frequency_cm-1": case["K_frequency_cm-1"],
                "n_configs": case["n_configs"],
                "final_population": case["final_population"],
                "supercell_min_frequency_cm-1": case[
                    "supercell_min_frequency_cm-1"
                ],
                "imaginary_modes_below_minus_1_cm-1": case[
                    "imaginary_modes_below_minus_1_cm-1"
                ],
                "K_neighbourhood_rises": side_rises(case, sample_distances),
                "sources": {
                    "result": str(case["result_path"].resolve()),
                    "result_sha256": case["result_sha256"],
                    "acceptance": str(case["acceptance_path"].resolve()),
                    "acceptance_sha256": case["acceptance_sha256"],
                },
            }
        )

    common_x = np.linspace(-0.04, 0.04, 801)
    centered = []
    for case in cases:
        order = np.argsort(case["signed_q"])
        centered.append(
            np.interp(
                common_x,
                case["signed_q"][order],
                case["top_cm-1"][order] - case["K_frequency_cm-1"],
            )
        )
    summary = {
        "status": "fixed_smearing_lattice_temperature_comparison_generated",
        "scope": (
            "actual converged quantum-SSCHA spectra at two lattice temperatures "
            "with one shared finite-cell electronic operator"
        ),
        "control": {
            "fixed_smearing_degauss_Ry": float(next(iter(smearings))),
            "same_electronic_operator_temperature_K": 300,
            "lattice_temperatures_K": [300, 450],
            "fixed_cell": True,
            "thermal_lattice_or_SSCHA_background_changes": True,
        },
        "case_metrics": case_metrics,
        "comparison": {
            "K_frequency_shift_450_minus_300_cm-1": float(
                cases[1]["K_frequency_cm-1"] - cases[0]["K_frequency_cm-1"]
            ),
            "maximum_K_referenced_shape_difference_abs_d_le_0p04_cm-1": float(
                np.max(np.abs(centered[1] - centered[0]))
            ),
            "RMSE_K_referenced_shape_difference_abs_d_le_0p04_cm-1": float(
                np.sqrt(np.mean((centered[1] - centered[0]) ** 2))
            ),
        },
        "X0_separability_gate": {
            "status": x0_acceptance["status"],
            "passes_X0_cross_gate": x0_acceptance["passes_X0_cross_gate"],
            "metrics": x0_acceptance["metrics"],
            "source": str(args.x0_acceptance.resolve()),
            "source_sha256": sha256(args.x0_acceptance),
            "interpretation": (
                "the plotted curves are actual SSCHA results; the failed gate forbids "
                "using simple diagonal-operator recombination as a general predictor"
            ),
        },
        "limitations": [
            (
                "this comparison uses the earlier finite-cell q6 electronic operator, "
                "not the newer dense-q full-EPC long-range response"
            ),
            (
                "only the 300-to-450 K pair is available at this fixed smearing; "
                "there is no controlled 600 K result at the same smearing"
            ),
            (
                "the comparison is a development result on the shared MLIP/SSCHA model, "
                "not an independent DFT-MD trajectory validation"
            ),
        ],
        "outputs": {"png": str(png_path.resolve())},
    }
    summary_path = args.output_dir / "fixed_smearing_lattice_temperature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
