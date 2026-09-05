#!/usr/bin/env python3
"""Compare the E0 EPW 450 K pilot with the converged static-DFPT line."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import friedel_module as fm  # noqa: E402
from graphene_fd_p4_common import (  # noqa: E402
    atomic_json,
    line_metrics,
    qpoint_fractional,
    sha256,
)


MEV_TO_CM1 = 8.06554393734921
THZ_TO_CM1 = 33.35640951981521

Q_RE = re.compile(
    r"ismear\s*=\s*\d+\s+iq\s*=\s*(\d+)\s+coord\.:\s*"
    r"([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+).*?Temp:\s*([-+0-9.Ee]+)K"
)
MODE_RE = re.compile(
    r"lambda___\(\s*(\d+)\s*\).*?omega=\s*([-+0-9.Ee]+)\s+meV\s+"
    r"on_shell_Omega=\s*([-+0-9.Ee]+)\s+meV\s+"
    r"off_shell_Omega=\s*([-+0-9.Ee]+)\s+meV"
)


def parse_epw_output(path: Path) -> tuple[list[dict], dict]:
    text = path.read_text(errors="replace").replace("\x00", "")
    qpoints: dict[int, dict] = {}
    current_iq = None
    for line in text.splitlines():
        match = Q_RE.search(line)
        if match:
            current_iq = int(match.group(1))
            qpoints[current_iq] = {
                "iq": current_iq,
                "q": [float(match.group(index)) for index in (2, 3, 4)],
                "temperature_K": float(match.group(5)),
                "modes": [],
            }
            continue
        match = MODE_RE.search(line)
        if match and current_iq is not None:
            qpoints[current_iq]["modes"].append(
                {
                    "mode": int(match.group(1)),
                    "bare_meV": float(match.group(2)),
                    "on_shell_meV": float(match.group(3)),
                    "off_shell_meV": float(match.group(4)),
                }
            )

    ordered = [qpoints[index] for index in sorted(qpoints)]
    if len(ordered) != 9 or any(len(point["modes"]) != 6 for point in ordered):
        raise ValueError(
            f"expected 9 q-points and 6 modes per point in {path}; "
            f"found {[len(point['modes']) for point in ordered]}"
        )

    def first_float(pattern: str):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    version = re.search(r"Program EPW v\.([^ ]+)", text)
    mesh = re.search(
        r"Using uniform k-mesh:\s*(\d+)\s+(\d+)\s+(\d+)", text
    )
    metadata = {
        "epw_version": version.group(1) if version else None,
        "completed": "Total program execution" in text,
        "fine_k_mesh": [int(mesh.group(index)) for index in (1, 2, 3)]
        if mesh
        else None,
        "target_temperature_eV": first_float(
            r"Golden Rule strictly enforced with T =\s*([-+0-9.Ee]+)\s+eV"
        ),
        "static_reference_smearing_Ry": first_float(
            r"static self-energy: T_high =\s*([-+0-9.Ee]+)\s+Ry"
        ),
        "gaussian_broadening_eV": first_float(
            r"Gaussian Broadening:\s*([-+0-9.Ee]+)\s+eV"
        ),
    }
    return ordered, metadata


def load_dfpt(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for region in ("G", "K"):
        selected = [row for row in rows if row["region"] == region]
        result[region] = {
            "t": np.asarray([float(row["t_GK"]) for row in selected], float),
            "frequencies": np.asarray(
                [
                    [float(row[f"f{mode}_cm-1"]) for mode in range(1, 7)]
                    for row in selected
                ],
                float,
            ),
        }
    return result


def interpolate_dfpt(reference, region: str, t: float) -> np.ndarray:
    return np.asarray(
        [
            np.interp(t, reference[region]["t"], reference[region]["frequencies"][:, mode])
            for mode in range(6)
        ],
        float,
    )


def load_static_reference(path: Path, t_values: np.ndarray) -> np.ndarray:
    phonon = fm.load_ph(path)
    phonon.run_qpoints(qpoint_fractional(t_values))
    frequencies = np.sort(
        np.asarray(phonon.get_qpoints_dict()["frequencies"], float), axis=1
    )
    return frequencies * THZ_TO_CM1


def signed_sqrt(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return np.sign(values) * np.sqrt(np.abs(values))


def metrics_for_variant(records: list[dict], key: str) -> tuple[dict, bool]:
    result = {}
    passes = True
    for region in ("G", "K"):
        selected = [record for record in records if record["region"] == region]
        t = np.asarray([record["t_GK"] for record in selected], float)
        prediction = np.asarray([record[key] for record in selected], float)
        target = np.asarray([record["target_top_cm-1"] for record in selected], float)
        metrics = line_metrics(region, t, prediction, target)
        result[region] = metrics
        passes = passes and metrics["line_MAE_cm-1"] < 10.0
        passes = passes and metrics["high_symmetry_abs_error_cm-1"] < 15.0
        if region == "K":
            passes = passes and metrics["kink_relative_error"] < 0.20
    return result, bool(passes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epw-output", type=Path, required=True)
    parser.add_argument("--dfpt-450", type=Path, required=True)
    parser.add_argument("--static-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    qpoints, metadata = parse_epw_output(args.epw_output)
    dfpt = load_dfpt(args.dfpt_450)
    t_values = np.asarray([3.0 * point["q"][0] for point in qpoints], float)
    static_reference = load_static_reference(args.static_reference, t_values)

    records = []
    for point_index, (point, t_value) in enumerate(zip(qpoints, t_values, strict=True)):
        region = "G" if t_value < 0.2 else "K"
        target = interpolate_dfpt(dfpt, region, float(t_value))
        modes = sorted(point["modes"], key=lambda row: row["mode"])
        bare = np.asarray([row["bare_meV"] for row in modes], float) * MEV_TO_CM1
        on_shell = np.asarray([row["on_shell_meV"] for row in modes], float) * MEV_TO_CM1
        off_shell = np.asarray([row["off_shell_meV"] for row in modes], float) * MEV_TO_CM1
        high = static_reference[point_index]
        transferred_on = signed_sqrt(high * high + on_shell * on_shell - bare * bare)
        transferred_off = signed_sqrt(high * high + off_shell * off_shell - bare * bare)
        for mode in range(6):
            records.append(
                {
                    "iq": int(point["iq"]),
                    "region": region,
                    "t_GK": float(t_value),
                    "qx_crystal": float(point["q"][0]),
                    "qy_crystal": float(point["q"][1]),
                    "mode": mode + 1,
                    "target_cm-1": float(target[mode]),
                    "static_reference_0.020Ry_cm-1": float(high[mode]),
                    "epw_bare_cm-1": float(bare[mode]),
                    "epw_on_shell_cm-1": float(on_shell[mode]),
                    "epw_off_shell_cm-1": float(off_shell[mode]),
                    "transferred_on_shell_cm-1": float(transferred_on[mode]),
                    "transferred_off_shell_cm-1": float(transferred_off[mode]),
                }
            )

    top_records = []
    for iq in range(1, 10):
        selected = [record for record in records if record["iq"] == iq]
        row = selected[-1]
        top_records.append(
            {
                "iq": iq,
                "region": row["region"],
                "t_GK": row["t_GK"],
                "target_top_cm-1": row["target_cm-1"],
                "static_reference_top_cm-1": row["static_reference_0.020Ry_cm-1"],
                "epw_bare_top_cm-1": row["epw_bare_cm-1"],
                "epw_on_shell_top_cm-1": row["epw_on_shell_cm-1"],
                "epw_off_shell_top_cm-1": row["epw_off_shell_cm-1"],
                "transferred_on_shell_top_cm-1": row["transferred_on_shell_cm-1"],
                "transferred_off_shell_top_cm-1": row["transferred_off_shell_cm-1"],
                "target_correction_from_0.020Ry_cm-1": row["target_cm-1"]
                - row["static_reference_0.020Ry_cm-1"],
                "epw_on_shell_correction_cm-1": row["epw_on_shell_cm-1"]
                - row["epw_bare_cm-1"],
                "epw_off_shell_correction_cm-1": row["epw_off_shell_cm-1"]
                - row["epw_bare_cm-1"],
            }
        )

    variant_keys = (
        "static_reference_top_cm-1",
        "epw_bare_top_cm-1",
        "epw_on_shell_top_cm-1",
        "epw_off_shell_top_cm-1",
        "transferred_on_shell_top_cm-1",
        "transferred_off_shell_top_cm-1",
    )
    all_metrics = {}
    numerical_gates = {}
    for key in variant_keys:
        all_metrics[key], numerical_gates[key] = metrics_for_variant(top_records, key)

    correction_metrics = {}
    for region in ("G", "K"):
        selected = [row for row in top_records if row["region"] == region]
        target = np.asarray(
            [row["target_correction_from_0.020Ry_cm-1"] for row in selected], float
        )
        correction_metrics[region] = {}
        for variant in ("on_shell", "off_shell"):
            predicted = np.asarray(
                [row[f"epw_{variant}_correction_cm-1"] for row in selected], float
            )
            correction_metrics[region][variant] = {
                "MAE_cm-1": float(np.mean(np.abs(predicted - target))),
                "RMSE_cm-1": float(np.sqrt(np.mean((predicted - target) ** 2))),
                "max_abs_cm-1": float(np.max(np.abs(predicted - target))),
            }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "epw_mode_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "epw_top_branch_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(top_records[0]))
        writer.writeheader()
        writer.writerows(top_records)

    summary = {
        "status": "completed",
        "scope": "EPW v6.0 dynamic phonon-self-energy restart pilot at 450 K",
        "metadata": metadata,
        "runtime": {
            "started_at": (args.epw_output.parent / "STARTED_AT").read_text().strip(),
            "completed_at": (args.epw_output.parent / "COMPLETED_AT").read_text().strip(),
        },
        "mesh": {"fine_k": metadata["fine_k_mesh"], "n_qpoints": 9},
        "metrics": all_metrics,
        "top_branch_numerical_gates": numerical_gates,
        "correction_metrics": correction_metrics,
        "passes_static_operator_gate": False,
        "static_operator_gate_reason": (
            "phonselfen returns dynamic on/off-shell frequency renormalization; "
            "the required adiabatic Pi(q,0,T)-Pi(q,0,T_ref) matrix was not exported"
        ),
        "next_stage": (
            "Do not expand temperatures yet. Export or implement the static Wannier "
            "band sum on the same q points, then reapply the 10/15 cm-1 and K-kink gates."
        ),
        "sources": [
            {"path": str(args.epw_output), "sha256": sha256(args.epw_output)},
            {"path": str(args.dfpt_450), "sha256": sha256(args.dfpt_450)},
            {
                "path": str(args.static_reference),
                "sha256": sha256(args.static_reference),
            },
        ],
    }
    atomic_json(output / "epw_pilot_summary.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, region in zip(axes, ("G", "K"), strict=True):
        selected = [row for row in top_records if row["region"] == region]
        t = np.asarray([row["t_GK"] for row in selected], float)
        axis.plot(t, [row["target_top_cm-1"] for row in selected], "o-", label="DFPT 450 K")
        axis.plot(
            t,
            [row["static_reference_top_cm-1"] for row in selected],
            "s--",
            label="static reference 0.020 Ry",
        )
        axis.plot(
            t,
            [row["transferred_on_shell_top_cm-1"] for row in selected],
            "^--",
            label="EPW dynamic correction (on-shell)",
        )
        axis.plot(
            t,
            [row["transferred_off_shell_top_cm-1"] for row in selected],
            "v--",
            label="EPW dynamic correction (off-shell)",
        )
        axis.set_title(f"{region} region")
        axis.set_xlabel("t along Γ–K")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("top branch (cm$^{-1}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    fig.savefig(output / "epw_pilot_top_branch.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
