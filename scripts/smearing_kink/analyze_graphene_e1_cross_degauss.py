#!/usr/bin/env python3
"""Evaluate cross-degauss DFT force differences against the E0 operators."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read


TEMPERATURES = (300, 450, 600)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(array, float) ** 2)))


def minimum_image_displacement(
    positions: np.ndarray, reference: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    fractional = (np.asarray(positions, float) - reference) @ np.linalg.inv(cell)
    fractional[:, :2] -= np.rint(fractional[:, :2])
    return fractional @ cell


def operator_force(force_constants: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    return -np.einsum("ijab,jb->ia", force_constants, displacement, optimize=True)


def group_metrics(records: list[dict]) -> dict:
    raw = np.concatenate([record["raw_difference"].ravel() for record in records])
    predicted = np.concatenate([record["predicted_difference"].ravel() for record in records])
    residual = raw - predicted
    raw_rmse = rms(raw)
    residual_rmse = rms(residual)
    relative = residual_rmse / raw_rmse if raw_rmse > 0.0 else float("inf")
    return {
        "n_tasks": len(records),
        "n_force_components": int(raw.size),
        "raw_difference_RMSE_meV_A": 1000.0 * raw_rmse,
        "predicted_difference_RMSE_meV_A": 1000.0 * rms(predicted),
        "residual_RMSE_meV_A": 1000.0 * residual_rmse,
        "residual_to_raw_RMSE": relative,
        "raw_predicted_component_correlation": (
            float(np.corrcoef(raw, predicted)[0, 1])
            if np.std(raw) > 0.0 and np.std(predicted) > 0.0
            else None
        ),
        "residual_max_abs_meV_A": 1000.0 * float(np.max(np.abs(residual))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--lane-a", type=Path, required=True)
    parser.add_argument("--lane-b", type=Path, required=True)
    parser.add_argument("--operator-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "frozen_not_released":
        raise ValueError("unexpected E1 manifest status")
    if manifest.get("locked_validation_temperatures_K_not_accessed") != [375, 525]:
        raise ValueError("locked validation boundary is missing")
    if sha256(args.configs) != manifest["configs"]["sha256"]:
        raise ValueError("E1 configuration hash mismatch")
    for lane, directory in (("A", args.lane_a), ("B", args.lane_b)):
        if not (directory / "DONE").is_file():
            raise FileNotFoundError(f"lane {lane} is not complete")
        summary = json.loads((directory / "summary.json").read_text())
        if summary.get("status") != "complete" or summary.get("lane") != lane:
            raise ValueError(f"invalid lane {lane} summary")
        if summary.get("configs_sha256") != manifest["configs"]["sha256"]:
            raise ValueError(f"lane {lane} used a different configuration file")

    decision_path = args.operator_dir / "operator_gate_decision.json"
    decision = json.loads(decision_path.read_text())
    if decision.get("status") != "passed" or decision.get("releases_E1") is not True:
        raise ValueError("operator aggregate does not release E1")
    if decision.get("development_temperatures_K") != list(TEMPERATURES):
        raise ValueError("unexpected operator temperatures")

    operators: dict[int, np.ndarray] = {}
    reference = cell = None
    operator_hashes = {}
    for entry in decision["operators"]:
        temperature = int(entry["temperature_K"])
        path = args.operator_dir / f"T{temperature}_operator.npz"
        if sha256(path) != entry["operator_sha256"]:
            raise ValueError(f"operator hash mismatch at T{temperature}")
        with np.load(path, allow_pickle=False) as data:
            force_constants = np.asarray(data["delta_fc_full"], float)
            current_reference = np.asarray(data["reference_positions"], float)
            current_cell = np.asarray(data["cell"], float)
            stored_temperature = float(data["temperature_K"])
        if force_constants.shape != (72, 72, 3, 3):
            raise ValueError(f"invalid force-constant shape at T{temperature}")
        if abs(stored_temperature - temperature) > 1.0e-8:
            raise ValueError(f"stored operator temperature mismatch at T{temperature}")
        if reference is None:
            reference, cell = current_reference, current_cell
        elif not (
            np.allclose(current_reference, reference, atol=1.0e-12, rtol=0.0)
            and np.allclose(current_cell, cell, atol=1.0e-12, rtol=0.0)
        ):
            raise ValueError("operator reference geometry changed with temperature")
        operators[temperature] = force_constants
        operator_hashes[str(temperature)] = entry["operator_sha256"]
    if set(operators) != set(TEMPERATURES) or reference is None or cell is None:
        raise ValueError("operator bundle is incomplete")

    frames = read(args.configs, ":")
    by_id = {str(frame.info["e1_config_id"]): frame for frame in frames}
    if len(by_id) != 15:
        raise ValueError("expected 15 unique E1 configurations")

    task_records = []
    for task in manifest["tasks"]:
        config_id = task["config_id"]
        frame = by_id[config_id]
        lane_dir = args.lane_a if task["lane"] == "A" else args.lane_b
        label_path = lane_dir / "labels" / f"{task['task_id']}.npz"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        with np.load(label_path, allow_pickle=False) as data:
            cross_forces = np.asarray(data["forces"], float)
            positions = np.asarray(data["positions"], float)
            label_cell = np.asarray(data["cell"], float)
            label_config = str(data["config_id"])
            target_degauss = float(data["degauss_Ry"])
        if label_config != config_id or cross_forces.shape != (72, 3):
            raise ValueError(f"invalid label identity or shape in {label_path}")
        if abs(target_degauss - float(task["target_degauss_Ry"])) > 1.0e-12:
            raise ValueError(f"degauss mismatch in {label_path}")
        if not np.allclose(positions, frame.positions, atol=1.0e-10, rtol=0.0):
            raise ValueError(f"position mismatch in {label_path}")
        if not (
            np.allclose(label_cell, frame.cell.array, atol=1.0e-10, rtol=0.0)
            and np.allclose(label_cell, cell, atol=1.0e-10, rtol=0.0)
        ):
            raise ValueError(f"cell mismatch in {label_path}")

        diagonal_temperature = int(round(float(frame.info["lattice_temperature_K"])))
        target_temperature = int(task["target_degauss_label"][1:])
        if diagonal_temperature == target_temperature:
            raise ValueError(f"task {task['task_id']} is not off diagonal")
        diagonal_forces = np.asarray(frame.arrays["REF_forces"], float)
        displacement = minimum_image_displacement(frame.positions, reference, cell)
        diagonal_operator_force = operator_force(
            operators[diagonal_temperature], displacement
        )
        target_operator_force = operator_force(operators[target_temperature], displacement)
        raw_difference = cross_forces - diagonal_forces
        predicted_difference = target_operator_force - diagonal_operator_force
        residual = raw_difference - predicted_difference
        task_records.append(
            {
                "task": task,
                "label_path": label_path,
                "diagonal_temperature_K": diagonal_temperature,
                "target_temperature_K": target_temperature,
                "displacement_max_A": float(
                    np.max(np.linalg.norm(displacement, axis=1))
                ),
                "raw_difference": raw_difference,
                "predicted_difference": predicted_difference,
                "residual": residual,
            }
        )
    if len(task_records) != 30:
        raise ValueError(f"expected 30 tasks, found {len(task_records)}")

    overall = group_metrics(task_records)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in task_records:
        grouped[
            f"T{record['diagonal_temperature_K']}_to_T{record['target_temperature_K']}"
        ].append(record)
    by_temperature_pair = {
        key: group_metrics(records) for key, records in sorted(grouped.items())
    }

    threshold = manifest["force_residual_gate"]
    checks = {
        "residual_RMSE_le_10_meV_A": (
            overall["residual_RMSE_meV_A"]
            <= float(threshold["residual_RMSE_meV_A_max"])
        ),
        "raw_le_10_meV_A_or_residual_le_20pct_raw": (
            overall["raw_difference_RMSE_meV_A"]
            <= float(threshold["residual_RMSE_meV_A_max"])
            or overall["residual_to_raw_RMSE"]
            <= float(threshold["relative_to_raw_max"])
        ),
    }
    passed = all(checks.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    component_path = args.output_dir / "force_component_records.csv"
    with component_path.open("w", newline="") as handle:
        fieldnames = [
            "task_id",
            "config_id",
            "lane",
            "diagonal_temperature_K",
            "target_temperature_K",
            "atom",
            "component",
            "raw_DFT_difference_eV_A",
            "operator_predicted_difference_eV_A",
            "residual_eV_A",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in task_records:
            for atom in range(72):
                for component, axis in enumerate("xyz"):
                    writer.writerow(
                        {
                            "task_id": record["task"]["task_id"],
                            "config_id": record["task"]["config_id"],
                            "lane": record["task"]["lane"],
                            "diagonal_temperature_K": record[
                                "diagonal_temperature_K"
                            ],
                            "target_temperature_K": record["target_temperature_K"],
                            "atom": atom,
                            "component": axis,
                            "raw_DFT_difference_eV_A": record["raw_difference"][
                                atom, component
                            ],
                            "operator_predicted_difference_eV_A": record[
                                "predicted_difference"
                            ][atom, component],
                            "residual_eV_A": record["residual"][atom, component],
                        }
                    )

    task_summary_path = args.output_dir / "task_summary.csv"
    with task_summary_path.open("w", newline="") as handle:
        fieldnames = [
            "task_id",
            "config_id",
            "lane",
            "diagonal_temperature_K",
            "target_temperature_K",
            "displacement_max_A",
            "raw_difference_RMSE_meV_A",
            "predicted_difference_RMSE_meV_A",
            "residual_RMSE_meV_A",
            "residual_to_raw_RMSE",
            "residual_max_abs_meV_A",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in task_records:
            metrics = group_metrics([record])
            writer.writerow(
                {
                    "task_id": record["task"]["task_id"],
                    "config_id": record["task"]["config_id"],
                    "lane": record["task"]["lane"],
                    "diagonal_temperature_K": record["diagonal_temperature_K"],
                    "target_temperature_K": record["target_temperature_K"],
                    "displacement_max_A": record["displacement_max_A"],
                    **{
                        key: metrics[key]
                        for key in fieldnames
                        if key in metrics
                    },
                }
            )

    payload = {
        "status": "passed" if passed else "failed",
        "scope": "E1 cross-degauss force residual gate on development temperatures",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_temperatures_K": list(TEMPERATURES),
        "locked_validation_temperatures_K_not_accessed": [375, 525],
        "formula": (
            "Delta F_pred(T_target,T_diag) = -[Delta Phi(T_target)-"
            "Delta Phi(T_diag)] u"
        ),
        "overall": overall,
        "by_temperature_pair": by_temperature_pair,
        "thresholds": threshold,
        "checks": checks,
        "decision": (
            "set_local_Mermin_term_to_zero"
            if passed
            else "run_E2_integrable_local_Mermin_free_energy_model"
        ),
        "sources": {
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "configs": {"path": str(args.configs), "sha256": sha256(args.configs)},
            "operator_decision": {
                "path": str(decision_path),
                "sha256": sha256(decision_path),
            },
            "operator_sha256_by_temperature": operator_hashes,
            "lane_A_summary_sha256": sha256(args.lane_a / "summary.json"),
            "lane_B_summary_sha256": sha256(args.lane_b / "summary.json"),
        },
        "outputs": {
            "force_component_records": str(component_path),
            "task_summary": str(task_summary_path),
        },
    }
    decision_output = args.output_dir / "e1_residual_gate.json"
    atomic_text(decision_output, json.dumps(payload, indent=2) + "\n")
    marker = args.output_dir / (
        "E1_RESIDUAL_GATE_PASS" if passed else "E1_RESIDUAL_GATE_FAIL"
    )
    atomic_text(marker, payload["evaluated_at_utc"] + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
