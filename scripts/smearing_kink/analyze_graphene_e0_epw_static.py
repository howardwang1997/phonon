#!/usr/bin/env python3
"""Evaluate the zero-frequency EPW self-energy difference against static DFPT."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_graphene_e0_epw_pilot import (
    MEV_TO_CM1,
    interpolate_dfpt,
    load_dfpt,
    load_static_reference,
    metrics_for_variant,
    parse_epw_output,
    signed_sqrt,
)
from graphene_fd_p4_common import atomic_json, sha256


def load_zero_frequency_self_energy(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 9:
            raise ValueError(f"unexpected specfun_sup.phon row: {line}")
        iq, mode = int(fields[0]), int(fields[1])
        values = [float(value) for value in fields[2:]]
        if abs(values[3]) <= 1.0e-12:
            rows.append(
                {
                    "iq": iq,
                    "mode": mode,
                    "temperature_K": values[0],
                    "numerical_broadening_eV": values[1],
                    "bare_eV": values[2],
                    "frequency_eV": values[3],
                    "pi_low_meV": values[4],
                    "pi_high_meV": values[5],
                    "imag_pi_low_meV": values[6],
                }
            )
    if len(rows) != 54:
        raise ValueError(f"expected 54 zero-frequency rows in {path}, found {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epw-output", type=Path, required=True)
    parser.add_argument("--static-self-energy", type=Path, required=True)
    parser.add_argument("--dfpt-450", type=Path)
    parser.add_argument("--dfpt-target", type=Path)
    parser.add_argument("--target-temperature-K", type=float, default=450.0)
    parser.add_argument("--static-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if (args.dfpt_450 is None) == (args.dfpt_target is None):
        parser.error("provide exactly one of --dfpt-450 and --dfpt-target")
    dfpt_path = args.dfpt_target or args.dfpt_450
    assert dfpt_path is not None
    target_temperature = float(args.target_temperature_K)
    qpoints, metadata = parse_epw_output(args.epw_output)
    self_energy = load_zero_frequency_self_energy(args.static_self_energy)
    dfpt = load_dfpt(dfpt_path)
    temperatures = {round(float(row["temperature_K"]), 8) for row in self_energy}
    if temperatures != {round(target_temperature, 8)}:
        raise ValueError(
            f"expected only {target_temperature} K rows, found {sorted(temperatures)}"
        )
    t_values = np.asarray([3.0 * point["q"][0] for point in qpoints], float)
    high_reference = load_static_reference(args.static_reference, t_values)
    self_energy_lookup = {
        (row["iq"], row["mode"]): row for row in self_energy
    }

    mode_records = []
    for point_index, (point, t_value) in enumerate(zip(qpoints, t_values, strict=True)):
        region = "G" if t_value < 0.2 else "K"
        target = interpolate_dfpt(dfpt, region, float(t_value))
        for mode in range(1, 7):
            row = self_energy_lookup[(point["iq"], mode)]
            bare_meV = row["bare_eV"] * 1000.0
            delta_pi_meV = row["pi_low_meV"] - row["pi_high_meV"]
            static_squared_meV2 = bare_meV * bare_meV + 2.0 * bare_meV * delta_pi_meV
            static_meV = float(signed_sqrt(np.asarray([static_squared_meV2]))[0])
            bare_cm1 = bare_meV * MEV_TO_CM1
            static_cm1 = static_meV * MEV_TO_CM1
            high_cm1 = float(high_reference[point_index, mode - 1])
            transferred_squared = high_cm1 * high_cm1 + static_cm1 * static_cm1 - bare_cm1 * bare_cm1
            transferred_cm1 = float(signed_sqrt(np.asarray([transferred_squared]))[0])
            mode_records.append(
                {
                    "iq": int(point["iq"]),
                    "region": region,
                    "t_GK": float(t_value),
                    "mode": mode,
                    "temperature_K": row["temperature_K"],
                    "numerical_broadening_eV": row["numerical_broadening_eV"],
                    "target_cm-1": float(target[mode - 1]),
                    "static_reference_0.020Ry_cm-1": high_cm1,
                    "epw_bare_cm-1": bare_cm1,
                    "pi_low_w0_meV": row["pi_low_meV"],
                    "pi_high_w0_meV": row["pi_high_meV"],
                    "delta_pi_w0_meV": delta_pi_meV,
                    "epw_static_cm-1": static_cm1,
                    "transferred_static_cm-1": transferred_cm1,
                }
            )

    top_records = []
    for iq in range(1, 10):
        row = [record for record in mode_records if record["iq"] == iq][-1]
        top_records.append(
            {
                "iq": iq,
                "region": row["region"],
                "t_GK": row["t_GK"],
                "target_top_cm-1": row["target_cm-1"],
                "static_reference_top_cm-1": row["static_reference_0.020Ry_cm-1"],
                "epw_bare_top_cm-1": row["epw_bare_cm-1"],
                "epw_static_top_cm-1": row["epw_static_cm-1"],
                "transferred_static_top_cm-1": row["transferred_static_cm-1"],
                "target_correction_from_0.020Ry_cm-1": row["target_cm-1"]
                - row["static_reference_0.020Ry_cm-1"],
                "epw_static_correction_cm-1": row["epw_static_cm-1"]
                - row["epw_bare_cm-1"],
                "delta_pi_w0_meV": row["delta_pi_w0_meV"],
            }
        )

    metrics = {}
    gates = {}
    for key in (
        "static_reference_top_cm-1",
        "epw_bare_top_cm-1",
        "epw_static_top_cm-1",
        "transferred_static_top_cm-1",
    ):
        metrics[key], gates[key] = metrics_for_variant(top_records, key)

    correction_metrics = {}
    for region in ("G", "K"):
        selected = [row for row in top_records if row["region"] == region]
        target = np.asarray(
            [row["target_correction_from_0.020Ry_cm-1"] for row in selected], float
        )
        predicted = np.asarray(
            [row["epw_static_correction_cm-1"] for row in selected], float
        )
        correction_metrics[region] = {
            "MAE_cm-1": float(np.mean(np.abs(predicted - target))),
            "RMSE_cm-1": float(np.sqrt(np.mean((predicted - target) ** 2))),
            "max_abs_cm-1": float(np.max(np.abs(predicted - target))),
        }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "static_mode_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mode_records[0]))
        writer.writeheader()
        writer.writerows(mode_records)
    with (output / "static_top_branch_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(top_records[0]))
        writer.writeheader()
        writer.writerows(top_records)

    passes_spectral_gate = gates["transferred_static_top_cm-1"]
    summary = {
        "status": "passed" if passes_spectral_gate else "failed",
        "scope": (
            "EPW zero-frequency static self-energy pilot at "
            f"{target_temperature:g} K"
        ),
        "target_temperature_K": target_temperature,
        "formula": (
            "Omega_static^2 = omega_ref^2 + 2*omega_epw*"
            f"[Pi(q,0,T={target_temperature:g}K)-Pi(q,0,T_high=0.020Ry)]"
        ),
        "metadata": metadata,
        "mesh": {"fine_k": metadata["fine_k_mesh"], "n_qpoints": 9},
        "metrics": metrics,
        "top_branch_numerical_gates": gates,
        "correction_metrics": correction_metrics,
        "passes_static_spectral_gate": passes_spectral_gate,
        "passes_replayable_force_operator_gate": False,
        "operator_gate_reason": (
            "This pilot exports mode-diagonal Pi on nine q points only; a Hermitian "
            "matrix on a complete commensurate q mesh and real-space replay are still required."
        ),
        "next_stage": (
            "If the spectral gate passes, converge the numerical broadening and k mesh, "
            "then export a complete q mesh for the Hermitian real-space FC2 construction."
            if passes_spectral_gate
            else "Do not expand to 300/600 K or force labels; first resolve the failed static spectral gate."
        ),
        "sources": [
            {"path": str(args.epw_output), "sha256": sha256(args.epw_output)},
            {
                "path": str(args.static_self_energy),
                "sha256": sha256(args.static_self_energy),
            },
            {"path": str(dfpt_path), "sha256": sha256(dfpt_path)},
            {
                "path": str(args.static_reference),
                "sha256": sha256(args.static_reference),
            },
        ],
    }
    atomic_json(output / "static_pilot_summary.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, region in zip(axes, ("G", "K"), strict=True):
        selected = [row for row in top_records if row["region"] == region]
        t = np.asarray([row["t_GK"] for row in selected], float)
        axis.plot(
            t,
            [row["target_top_cm-1"] for row in selected],
            "o-",
            label=f"DFPT {target_temperature:g} K",
        )
        axis.plot(
            t,
            [row["static_reference_top_cm-1"] for row in selected],
            "s--",
            label="static reference 0.020 Ry",
        )
        axis.plot(
            t,
            [row["transferred_static_top_cm-1"] for row in selected],
            "^--",
            label="EPW static Pi transfer",
        )
        axis.set_title(f"{region} region")
        axis.set_xlabel("t along Γ–K")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("top branch (cm$^{-1}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    fig.savefig(output / "static_pilot_top_branch.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
