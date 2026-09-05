#!/usr/bin/env python3
"""Classify historical X0 DFT geometries against the corrected X0 support.

Only positions, cells, the folded-K A' eigenvector, and the corrected 300-point
ensemble are read.  No DFT energy or force enters the support classification.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from audit_graphene_current_s0_fixed_smearing import (  # noqa: E402
    folded_k_aprime_mode,
    load_operator,
    minimum_image_vectors,
    sha256,
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def displacement_from_reference(
    positions: np.ndarray, reference: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    values = []
    for frame in positions:
        vectors = minimum_image_vectors(frame, reference, cell)
        diagonal = vectors[np.arange(len(reference)), np.arange(len(reference))]
        values.append(diagonal)
    return np.asarray(values)


def minimum_pair_distance(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(cell)
    output = []
    for frame in positions:
        vectors = frame[:, None, :] - frame[None, :, :]
        fractional = vectors @ inverse
        fractional -= np.round(fractional)
        distances = np.linalg.norm(fractional @ cell, axis=2)
        distances += np.eye(len(frame)) * 1.0e9
        output.append(float(np.min(distances)))
    return np.asarray(output)


def physical_features(
    positions: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray,
    aprime_mode: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = displacement_from_reference(positions, reference, cell)
    mode = aprime_mode.reshape(-1)
    coordinate = np.asarray(
        [np.vdot(mode, value.reshape(-1)) for value in displacement]
    )
    features = np.column_stack(
        [
            np.sqrt(np.mean(displacement**2, axis=(1, 2))),
            np.max(np.linalg.norm(displacement, axis=2), axis=1),
            np.sqrt(np.mean(displacement[:, :, :2] ** 2, axis=(1, 2))),
            np.sqrt(np.mean(displacement[:, :, 2] ** 2, axis=1)),
            np.abs(coordinate),
            minimum_pair_distance(positions, cell),
        ]
    )
    return features, displacement


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-selected", type=Path, required=True)
    parser.add_argument("--corrected-xats", type=Path, required=True)
    parser.add_argument("--corrected-acceptance", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--corrected-result", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--core-percentile", type=float, default=95.0)
    parser.add_argument("--support-percentile", type=float, default=99.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    historical = read(args.historical_selected, index=":")
    historical_positions = np.asarray([atoms.positions for atoms in historical])
    corrected_cc = np.asarray(np.load(args.corrected_xats), float)
    acceptance = json.loads(args.corrected_acceptance.read_text(encoding="utf-8"))
    interface = acceptance.get("atom_order_interface", {})
    if interface.get("status") != "validated_before_SSCHA":
        raise ValueError("corrected X0 did not validate its atom-order interface")
    phonopy_for_cc = np.asarray(interface["phonopy_for_CellConstructor"], int)
    cc_for_phonopy = np.argsort(phonopy_for_cc)
    corrected_positions = corrected_cc[:, cc_for_phonopy]

    _, reference, cell, _ = load_operator(args.operator)
    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.corrected_result, reference, cell
    )
    corrected_features, corrected_displacement = physical_features(
        corrected_positions, reference, cell, mode
    )
    historical_features, historical_displacement = physical_features(
        historical_positions, reference, cell, mode
    )
    feature_names = np.asarray(
        [
            "RMS_displacement_A",
            "max_displacement_A",
            "inplane_RMS_displacement_A",
            "outplane_RMS_displacement_A",
            "Aprime_coordinate_abs_A",
            "minimum_pair_distance_A",
        ]
    )
    median = np.median(corrected_features, axis=0)
    lower = np.percentile(corrected_features, 25.0, axis=0)
    upper = np.percentile(corrected_features, 75.0, axis=0)
    scale = upper - lower
    scale = np.where(scale > 1.0e-12, scale, np.std(corrected_features, axis=0))
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    corrected_standard = (corrected_features - median) / scale
    historical_standard = (historical_features - median) / scale

    if args.neighbors < 1 or args.neighbors >= len(corrected_standard):
        raise ValueError("invalid nearest-neighbor count")
    self_distance = cdist(corrected_standard, corrected_standard)
    np.fill_diagonal(self_distance, np.inf)
    corrected_knn = np.mean(
        np.partition(self_distance, args.neighbors - 1, axis=1)[:, : args.neighbors],
        axis=1,
    )
    query_distance = cdist(historical_standard, corrected_standard)
    historical_knn = np.mean(
        np.partition(query_distance, args.neighbors - 1, axis=1)[:, : args.neighbors],
        axis=1,
    )
    core_threshold = float(np.percentile(corrected_knn, args.core_percentile))
    support_threshold = float(np.percentile(corrected_knn, args.support_percentile))
    levels = np.where(
        historical_knn <= core_threshold,
        "corrected_X0_core",
        np.where(
            historical_knn <= support_threshold,
            "corrected_X0_edge",
            "outside_corrected_X0_support",
        ),
    )
    percentile_rank = np.asarray(
        [100.0 * np.mean(corrected_knn <= value) for value in historical_knn]
    )

    records = []
    for index, structure in enumerate(historical):
        record = {
            "sscha_index": int(structure.info["sscha_index"]),
            "historical_selection_group": str(structure.info["selection_group"]),
            "corrected_X0_support_level": str(levels[index]),
            "mean_5NN_distance": float(historical_knn[index]),
            "corrected_leave_one_out_percentile_rank": float(percentile_rank[index]),
        }
        record.update(
            {
                str(name): float(value)
                for name, value in zip(feature_names, historical_features[index])
            }
        )
        records.append(record)

    counts = {
        level: int(np.sum(levels == level))
        for level in (
            "corrected_X0_core",
            "corrected_X0_edge",
            "outside_corrected_X0_support",
        )
    }
    summary = {
        "status": "geometry_only_support_frozen",
        "DFT_energy_or_force_read": False,
        "scope": "historical 12-point X0 selection classified against corrected 300-point X0 geometry support",
        "protocol": {
            "features": feature_names.tolist(),
            "standardization": "corrected-X0 median and interquartile range",
            "distance": "Euclidean in robust-standardized feature space",
            "neighbors": args.neighbors,
            "core_threshold": f"corrected leave-one-out {args.core_percentile:g}th percentile",
            "support_threshold": f"corrected leave-one-out {args.support_percentile:g}th percentile",
        },
        "thresholds": {
            "core_mean_5NN_distance": core_threshold,
            "support_mean_5NN_distance": support_threshold,
        },
        "counts": counts,
        "corrected_feature_ranges": {
            str(name): {
                "minimum": float(np.min(corrected_features[:, column])),
                "percentile_1": float(np.percentile(corrected_features[:, column], 1.0)),
                "median": float(np.median(corrected_features[:, column])),
                "percentile_99": float(np.percentile(corrected_features[:, column], 99.0)),
                "maximum": float(np.max(corrected_features[:, column])),
            }
            for column, name in enumerate(feature_names)
        },
        "Aprime_mode": mode_provenance,
        "inputs": {
            "historical_selected": {
                "path": str(args.historical_selected),
                "sha256": sha256(args.historical_selected),
            },
            "corrected_xats": {
                "path": str(args.corrected_xats),
                "sha256": sha256(args.corrected_xats),
            },
            "corrected_acceptance": {
                "path": str(args.corrected_acceptance),
                "sha256": sha256(args.corrected_acceptance),
            },
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
            "background": {
                "path": str(args.background),
                "sha256": sha256(args.background),
            },
            "corrected_result": {
                "path": str(args.corrected_result),
                "sha256": sha256(args.corrected_result),
            },
        },
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "geometry_support_summary.json", summary)
    with (args.output_dir / "geometry_support_records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    atomic_npz(
        args.output_dir / "geometry_support_diagnostics.npz",
        feature_names=feature_names,
        corrected_features=corrected_features,
        historical_features=historical_features,
        corrected_standardized_features=corrected_standard,
        historical_standardized_features=historical_standard,
        corrected_leave_one_out_5NN_distance=corrected_knn,
        historical_to_corrected_5NN_distance=historical_knn,
        historical_support_levels=levels,
        corrected_displacement_A=corrected_displacement,
        historical_displacement_A=historical_displacement,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "thresholds": summary["thresholds"],
                "counts": counts,
                "records": records,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
