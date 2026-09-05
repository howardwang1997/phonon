#!/usr/bin/env python3
"""Gate the first non-diagonal X0 condition and reconstruct the development grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import anomaly_locate as anomaly  # noqa: E402
import friedel_module as fm  # noqa: E402
import td_phonon as tdp  # noqa: E402


CM_PER_THZ = 33.35641


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verified_path(record: dict) -> Path:
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"frozen input hash changed: {path}")
    return path


def load_fc2(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        force_constants = np.asarray(data[key], float)
    if (
        force_constants.ndim != 4
        or force_constants.shape[0] != force_constants.shape[1]
        or force_constants.shape[2:] != (3, 3)
        or not np.isfinite(force_constants).all()
    ):
        raise ValueError(f"invalid FC2 {key} in {path}")
    return force_constants


def spectrum(phonon, force_constants: np.ndarray, npoints: int = 201):
    distance, frequency, label_positions, labels = tdp.band_from_phonopy(
        phonon, force_constants, npoints=npoints
    )
    distance = np.asarray(distance, float)
    frequency = np.asarray(frequency, float)
    if not np.isfinite(frequency).all():
        raise ValueError("non-finite X0 spectrum")
    return distance, frequency, np.asarray(label_positions, float), np.asarray(labels)


def cross_metrics(
    distance: np.ndarray,
    formula_frequency_thz: np.ndarray,
    coupled_frequency_thz: np.ndarray,
    label_positions: np.ndarray,
    labels: np.ndarray,
) -> dict:
    if formula_frequency_thz.shape != coupled_frequency_thz.shape:
        raise ValueError("formula and coupled spectra have different shapes")
    top_formula = np.asarray(formula_frequency_thz[:, -1], float)
    top_coupled = np.asarray(coupled_frequency_thz[:, -1], float)
    top_difference_cm = (top_coupled - top_formula) * CM_PER_THZ
    names = [str(label) for label in labels]
    point_differences = {}
    for label in (r"$\Gamma$", "K"):
        indices = [index for index, name in enumerate(names) if name == label]
        if not indices:
            raise ValueError(f"missing path label {label}")
        values = []
        for label_index in indices:
            point_index = int(np.argmin(np.abs(distance - label_positions[label_index])))
            values.append(float(top_difference_cm[point_index]))
        point_differences[label] = max(values, key=abs)

    formula_kinks = {
        item["label"]: item
        for item in anomaly.high_sym_kinks(
            distance, formula_frequency_thz, label_positions, labels
        )
    }
    coupled_kinks = {
        item["label"]: item
        for item in anomaly.high_sym_kinks(
            distance, coupled_frequency_thz, label_positions, labels
        )
    }
    formula_k = float(formula_kinks["K"]["kink_strength"])
    coupled_k = float(coupled_kinks["K"]["kink_strength"])
    relative_kink_change = abs(coupled_k - formula_k) / max(abs(formula_k), 1.0e-12)
    return {
        "top_branch_max_abs_difference_cm-1": float(np.max(np.abs(top_difference_cm))),
        "top_branch_RMSE_cm-1": float(np.sqrt(np.mean(top_difference_cm**2))),
        "high_symmetry_top_branch_difference_cm-1": point_differences,
        "formula_K_kink_THz_per_q": formula_k,
        "coupled_K_kink_THz_per_q": coupled_k,
        "K_kink_relative_change": float(relative_kink_change),
    }


def operator(freeze: dict, temperature: int) -> tuple[Path, np.ndarray]:
    record = freeze["source_L0"]["operators"][str(temperature)]
    path = verified_path(record)
    return path, load_fc2(path, "delta_fc_full")


def q0_result(freeze: dict, temperature: int) -> tuple[Path, np.ndarray, Path]:
    q0_root = Path(freeze["paths"]["q0_root"])
    result = q0_root / f"formal_T{temperature}" / "result.npz"
    acceptance_path = q0_root / f"formal_T{temperature}" / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    if acceptance.get("status") != "passed" or acceptance.get("converged") is not True:
        raise ValueError(f"Q0 T={temperature} did not pass")
    if acceptance.get("result_sha256") != sha256(result):
        raise ValueError(f"Q0 T={temperature} result hash mismatch")
    return result, load_fc2(result, "free_energy_fc2_eV_A2"), acceptance_path


def formula_grid(freeze: dict, phonon, output_dir: Path) -> dict:
    arrays = {}
    records = []
    for lattice_temperature in (300, 450, 600):
        diagonal_path, diagonal_fc2, acceptance_path = q0_result(
            freeze, lattice_temperature
        )
        _, diagonal_operator = operator(freeze, lattice_temperature)
        for operator_temperature in (300, 450, 600):
            operator_path, target_operator = operator(freeze, operator_temperature)
            reconstructed = diagonal_fc2 - diagonal_operator + target_operator
            distance, frequency, label_positions, labels = spectrum(phonon, reconstructed)
            prefix = f"Tlat{lattice_temperature}_Tel{operator_temperature}"
            arrays[f"{prefix}_fc2"] = reconstructed
            arrays[f"{prefix}_distance"] = distance
            arrays[f"{prefix}_frequency_cm-1"] = frequency * CM_PER_THZ
            arrays["label_positions"] = label_positions
            arrays["labels"] = labels
            records.append(
                {
                    "lattice_temperature_K": lattice_temperature,
                    "operator_temperature_K": operator_temperature,
                    "construction": (
                        "Q0 diagonal free-energy Hessian - diagonal electronic operator "
                        "+ target electronic operator"
                    ),
                    "diagonal_Q0_result": str(diagonal_path),
                    "diagonal_Q0_result_sha256": sha256(diagonal_path),
                    "diagonal_Q0_acceptance_sha256": sha256(acceptance_path),
                    "target_operator": str(operator_path),
                    "target_operator_sha256": sha256(operator_path),
                }
            )
    grid_path = output_dir / "formula_grid_3x3.npz"
    atomic_npz(grid_path, **arrays)
    return {
        "path": str(grid_path),
        "sha256": sha256(grid_path),
        "conditions": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("sscha", "md"), required=True)
    parser.add_argument("--actual-result", type=Path)
    parser.add_argument("--coupled-short-tdep", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze_manifest.read_text())
    if freeze.get("status") != "frozen_for_Q0_X0_development":
        raise ValueError("invalid Q0/X0 freeze manifest")
    if freeze.get("locked_validation_temperatures_K_not_accessed") != [375, 525]:
        raise ValueError("locked validation boundary is missing")
    if sha256(Path(__file__).resolve()) != freeze["scripts"]["x0_analysis"]["sha256"]:
        raise ValueError("X0 analysis script changed after the development protocol was frozen")
    condition = freeze["X0_protocol"]["first_non_diagonal_condition"]
    lattice_temperature = int(condition["lattice_temperature_K"])
    operator_temperature = int(condition["operator_temperature_K"])
    background_path = verified_path(freeze["source_L0"]["background"])
    phonon = fm.load_ph(background_path)
    diagonal_operator_path, diagonal_operator = operator(freeze, lattice_temperature)
    target_operator_path, target_operator = operator(freeze, operator_temperature)

    if args.mode == "sscha":
        if args.actual_result is None:
            parser.error("--actual-result is required in sscha mode")
        actual_acceptance_path = args.actual_result.parent / "acceptance.json"
        actual_acceptance = json.loads(actual_acceptance_path.read_text())
        if actual_acceptance.get("status") != "passed" or not actual_acceptance.get(
            "converged"
        ):
            raise ValueError("first non-diagonal SSCHA condition did not converge")
        if actual_acceptance.get("result_sha256") != sha256(args.actual_result):
            raise ValueError("X0 SSCHA result hash differs from acceptance record")
        diagonal_path, diagonal_fc2, diagonal_acceptance_path = q0_result(
            freeze, lattice_temperature
        )
        formula_fc2 = diagonal_fc2 - diagonal_operator + target_operator
        coupled_fc2 = load_fc2(args.actual_result, "free_energy_fc2_eV_A2")
        source_records = {
            "diagonal_Q0": {
                "path": str(diagonal_path),
                "sha256": sha256(diagonal_path),
                "acceptance_sha256": sha256(diagonal_acceptance_path),
            },
            "coupled_X0": {
                "path": str(args.actual_result),
                "sha256": sha256(args.actual_result),
                "acceptance_sha256": sha256(actual_acceptance_path),
            },
        }
    else:
        if args.coupled_short_tdep is None:
            parser.error("--coupled-short-tdep is required in md mode")
        baseline_record = freeze["source_L0"]["results"][str(lattice_temperature)][
            "seed0_short_tdep"
        ]
        baseline_path = verified_path(baseline_record)
        baseline_short = load_fc2(baseline_path, f"T{lattice_temperature}_fc2")
        coupled_short = load_fc2(
            args.coupled_short_tdep, f"T{lattice_temperature}_fc2"
        )
        formula_fc2 = baseline_short + target_operator
        coupled_fc2 = coupled_short + target_operator
        source_records = {
            "diagonal_L0_seed0_short_TDEP": baseline_record,
            "coupled_X0_seed0_short_TDEP": {
                "path": str(args.coupled_short_tdep),
                "sha256": sha256(args.coupled_short_tdep),
            },
        }

    formula_distance, formula_frequency, label_positions, labels = spectrum(
        phonon, formula_fc2
    )
    coupled_distance, coupled_frequency, coupled_label_positions, coupled_labels = spectrum(
        phonon, coupled_fc2
    )
    if not np.allclose(formula_distance, coupled_distance):
        raise ValueError("formula and coupled paths differ")
    if not np.allclose(label_positions, coupled_label_positions) or not np.array_equal(
        labels, coupled_labels
    ):
        raise ValueError("formula and coupled path labels differ")
    metrics = cross_metrics(
        formula_distance,
        formula_frequency,
        coupled_frequency,
        label_positions,
        labels,
    )
    thresholds = freeze["X0_protocol"]["fixed_gate"]
    line_pass = metrics["top_branch_max_abs_difference_cm-1"] < float(
        thresholds["top_branch_max_abs_difference_cm-1"]
    )
    kink_pass = metrics["K_kink_relative_change"] < float(
        thresholds["K_kink_relative_change"]
    )
    passed = bool(line_pass and kink_pass)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra_path = output_dir / f"{args.mode}_cross_comparison.npz"
    atomic_npz(
        spectra_path,
        distance=formula_distance,
        label_positions=label_positions,
        labels=labels,
        formula_frequency_cm_1=formula_frequency * CM_PER_THZ,
        coupled_frequency_cm_1=coupled_frequency * CM_PER_THZ,
        formula_fc2=formula_fc2,
        coupled_fc2=coupled_fc2,
    )
    grid = formula_grid(freeze, phonon, output_dir) if passed and args.mode == "sscha" else None
    payload = {
        "status": "passed" if passed else "cross_term_exceeds_fixed_gate",
        "mode": args.mode,
        "scope": "first fixed non-diagonal X0 development condition",
        "condition": {
            "lattice_temperature_K": lattice_temperature,
            "operator_temperature_K": operator_temperature,
            "operator_degauss_Ry": freeze["source_L0"]["operators"][
                str(operator_temperature)
            ]["degauss_Ry"],
            "operator_degauss_formula": "k_B*T_operator/Ry",
        },
        "formula": freeze["X0_protocol"]["formula_reconstruction"],
        "thresholds": {
            "top_branch_max_abs_difference_cm-1": thresholds[
                "top_branch_max_abs_difference_cm-1"
            ],
            "K_kink_relative_change": thresholds["K_kink_relative_change"],
        },
        "metrics": metrics,
        "passes_top_branch_gate": line_pass,
        "passes_K_kink_gate": kink_pass,
        "passes_X0_cross_gate": passed,
        "comparison_spectra": {"path": str(spectra_path), "sha256": sha256(spectra_path)},
        "formula_grid_3x3": grid,
        "inputs": {
            "freeze_manifest": {
                "path": str(args.freeze_manifest),
                "sha256": sha256(args.freeze_manifest),
            },
            "background": freeze["source_L0"]["background"],
            "diagonal_operator": {
                "path": str(diagonal_operator_path),
                "sha256": sha256(diagonal_operator_path),
            },
            "target_operator": {
                "path": str(target_operator_path),
                "sha256": sha256(target_operator_path),
            },
            **source_records,
        },
        "next_stage": (
            "freeze formula-reconstructed 3x3 development grid"
            if passed and args.mode == "sscha"
            else "run one fixed-seed classical-MD diagnostic, then stop"
            if args.mode == "sscha"
            else "stop before any automatic three-seed cross-term expansion"
        ),
    }
    acceptance_path = output_dir / f"{args.mode}_acceptance.json"
    atomic_json(acceptance_path, payload)

    if passed and args.mode == "sscha":
        q0_root = Path(freeze["paths"]["q0_root"])
        candidate = {
            "status": "ready_for_validation_freeze_review",
            "created_from_development_only": True,
            "locked_validation_temperatures_K_not_accessed": [375, 525],
            "automatic_validation_release": False,
            "Q0_all_temperatures_marker": str(q0_root / "Q0_ALL_TEMPERATURES_PASSED"),
            "Q0_summary_hashes": {
                str(temperature): sha256(q0_root / f"formal_T{temperature}/acceptance.json")
                for temperature in (300, 450, 600)
            },
            "X0_acceptance": {"path": str(acceptance_path), "sha256": sha256(acceptance_path)},
            "formula_grid_3x3": grid,
            "next_action": "review and explicitly freeze before generating 375/525 K validation data",
        }
        atomic_json(output_dir / "development_freeze_candidate.json", candidate)

    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
