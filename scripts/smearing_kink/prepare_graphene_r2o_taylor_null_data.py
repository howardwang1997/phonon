#!/usr/bin/env python3
"""Freeze the leakage-safe R2O Taylor-null training and gate data.

R2O is a force-only, finite-amplitude correction.  Its gradient data are the
92 thermal structures already frozen for R2M: exact E50 seed 0 plus the
matched-operator T300/T600 auxiliary structures.  E50 seed 1 is an endpoint
gate only.  The 25 harmonic-validation structures are split by a *fixed*
nearest-neighbour bond-length RMS threshold; the small-amplitude subset is
the physical harmonic force gate and all 25 remain a report-only diagnostic.

This program deliberately has no seed2 or support input and never walks the
source directory.  It reads only the three explicitly named R2M files whose
hashes are frozen below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list
from phonopy import Phonopy
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import td_common as tdc  # noqa: E402
from phonon_accel.phonons import ase_to_phonopy, phonopy_to_ase  # noqa: E402


FORMAT = "graphene_r2o_taylor_null_data_v1"
FIXED_DEGAUSS_RY = 0.0019000869380739254
MATCHED_T600_DEGAUSS_RY = 0.003800173876147851
R_MAX_A = 3.2
EDGE_BUFFER_A = 0.0
SMALL_BOND_RMS_MAX_A = 0.003
REFERENCE_LATTICE_A = 2.4600000087
EXPECTED_INPUT_HASHES = {
    "manifest.json": "bb20a86af073e344c38e39ab14d3a80af9e39fa051591bed2b039ae643ffe31e",
    "train.xyz": "789b65e1a2c3b260e24d80f008407c7ee55e02676ff3bcf9a875fe9de294519c",
    "valid.xyz": "92ad1508e1067265445b79679cfe5142d23b2507dab2aa44397578cee8c7a6da",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_extxyz(path: Path, structures: list[Atoms]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write(temporary, structures, format="extxyz")
    os.replace(temporary, path)


def geometry_fingerprint(structure: Atoms) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(structure.numbers, np.dtype("<i8")).tobytes())
    digest.update(np.round(np.asarray(structure.cell, float), 10).astype("<f8").tobytes())
    digest.update(
        np.round(np.asarray(structure.positions, float), 10).astype("<f8").tobytes()
    )
    return digest.hexdigest()


def graphene_reference(structure: Atoms) -> tuple[Atoms, int]:
    """Return the DFT/TDEP phonopy reference in its exact atom order.

    The tiny projection onto the source cell changes no ordering or fractional
    coordinate.  It makes the live Cartesian graph use one identical cell for
    ``x`` and ``x0`` rather than silently differentiating through a 5e-8 A
    lattice mismatch.
    """
    count = len(structure)
    size = int(round(np.sqrt(count / 2.0)))
    if count != 2 * size * size or size not in {6, 8}:
        raise ValueError(f"R2O supports only ordered 6x6/8x8 graphene; n={count}")
    if set(map(int, structure.numbers)) != {6}:
        raise ValueError("R2O reference is carbon-only")
    primitive = tdc.build_monolayer(
        "graphene", vacuum=7.5, a=REFERENCE_LATTICE_A
    )
    primitive.wrap()
    phonon = Phonopy(
        ase_to_phonopy(primitive),
        supercell_matrix=np.diag((size, size, 1)),
        primitive_matrix=np.eye(3),
    )
    reference = phonopy_to_ase(phonon.supercell)
    reference.wrap()
    if len(reference) != count or not np.array_equal(reference.numbers, structure.numbers):
        raise ValueError("phonopy reference atom order/species changed")
    source_cell = np.asarray(structure.cell, float)
    template_cell_error = float(
        np.max(np.abs(np.asarray(reference.cell, float) - source_cell))
    )
    if template_cell_error > 1.0e-5:
        raise ValueError(
            f"source cell differs from the frozen phonopy reference by {template_cell_error:g} A"
        )
    scaled = np.asarray(reference.get_scaled_positions(wrap=False), float)
    reference.set_cell(source_cell, scale_atoms=False)
    reference.set_scaled_positions(scaled)
    reference.pbc = np.asarray(structure.pbc, bool)
    reference.wrap()
    reference.info.update(
        {
            "r2o_reference": True,
            "r2o_supercell_size": size,
            "r2o_order": "phonopy_supercell_order_from_fixed_DFT_TDEP_builder",
            "r2o_reference_lattice_a_A": REFERENCE_LATTICE_A,
            "r2o_template_cell_projection_max_abs_A": template_cell_error,
        }
    )
    return reference, size


def minimum_image_displacement(structure: Atoms, reference: Atoms) -> np.ndarray:
    if len(structure) != len(reference):
        raise ValueError("reference atom count changed")
    cell = np.asarray(reference.cell, float)
    if not np.allclose(np.asarray(structure.cell, float), cell, atol=1.0e-10, rtol=0.0):
        raise ValueError("reference/current cells differ")
    fractional = (np.asarray(structure.positions) - np.asarray(reference.positions)) @ np.linalg.inv(cell)
    for axis, periodic in enumerate(reference.pbc):
        if periodic:
            fractional[:, axis] -= np.rint(fractional[:, axis])
    return fractional @ cell


def reference_assignment(structure: Atoms, reference: Atoms) -> tuple[np.ndarray, dict]:
    """Map reference indices to source indices with a species-aware MIC solve."""
    if len(structure) != len(reference):
        raise ValueError("assignment atom count changed")
    cell = np.asarray(reference.cell, float)
    if not np.allclose(np.asarray(structure.cell, float), cell, atol=1.0e-10, rtol=0.0):
        raise ValueError("assignment cells differ")
    delta = (
        np.asarray(structure.positions, float)[:, None, :]
        - np.asarray(reference.positions, float)[None, :, :]
    )
    fractional = delta @ np.linalg.inv(cell)
    for axis, periodic in enumerate(reference.pbc):
        if periodic:
            fractional[:, :, axis] -= np.rint(fractional[:, :, axis])
    distance = np.linalg.norm(fractional @ cell, axis=-1)
    incompatible = structure.numbers[:, None] != reference.numbers[None, :]
    cost = np.where(incompatible, 1.0e9, distance)
    source_index, reference_index = linear_sum_assignment(cost)
    if not np.array_equal(np.sort(reference_index), np.arange(len(reference))):
        raise ValueError("assignment does not cover every reference atom")
    reference_to_source = np.empty(len(reference), dtype=int)
    reference_to_source[reference_index] = source_index
    assigned = distance[reference_to_source, np.arange(len(reference))]
    sorted_distance = np.sort(distance, axis=1)
    nearest_gap = sorted_distance[:, 1] - sorted_distance[:, 0]
    nearest_reference = np.argmin(distance, axis=1)
    assigned_reference_for_source = np.empty(len(reference), dtype=int)
    assigned_reference_for_source[source_index] = reference_index
    unique_nearest = bool(
        np.array_equal(assigned_reference_for_source, nearest_reference)
        and float(np.min(nearest_gap)) > 1.0e-8
    )
    return reference_to_source, {
        "identity": bool(np.array_equal(reference_to_source, np.arange(len(reference)))),
        "unique_nearest": unique_nearest,
        "maximum_assigned_distance_A": float(np.max(assigned)),
        "minimum_nearest_second_nearest_gap_A": float(np.min(nearest_gap)),
    }


def force_metrics(structures: list[Atoms]) -> dict:
    joined = np.concatenate(
        [np.asarray(item.arrays["REF_forces"], float).reshape(-1) for item in structures]
    )
    return {
        "RMSE_meV_A": float(1000.0 * np.sqrt(np.mean(np.square(joined)))),
        "max_abs_meV_A": float(1000.0 * np.max(np.abs(joined))),
        "n_force_components": int(joined.size),
    }


def _canonical_edge_key(i: int, j: int, shift: np.ndarray) -> tuple:
    forward = (int(i), int(j), *(int(value) for value in shift))
    inverse = (int(j), int(i), *(-int(value) for value in shift))
    return min(forward, inverse)


def undirected_edges(structure: Atoms, cutoff: float) -> set[tuple]:
    i, j, shifts = neighbor_list("ijS", structure, cutoff, self_interaction=False)
    return {
        _canonical_edge_key(left, right, shift)
        for left, right, shift in zip(i, j, shifts, strict=True)
    }


def nearest_bond_keys(reference: Atoms) -> list[tuple]:
    keys = sorted(undirected_edges(reference, 1.8))
    expected = 3 * len(reference) // 2
    if len(keys) != expected:
        raise ValueError(f"expected {expected} nearest-neighbour bonds, got {len(keys)}")
    return keys


def edge_length(structure: Atoms, key: tuple) -> float:
    i, j, sx, sy, sz = key
    vector = (
        np.asarray(structure.positions[j], float)
        + np.asarray([sx, sy, sz], float) @ np.asarray(structure.cell, float)
        - np.asarray(structure.positions[i], float)
    )
    return float(np.linalg.norm(vector))


def bond_rms(structure: Atoms, reference: Atoms, bonds: list[tuple]) -> float:
    changes = np.asarray(
        [edge_length(structure, key) - edge_length(reference, key) for key in bonds]
    )
    return float(np.sqrt(np.mean(np.square(changes))))


def clone(structure: Atoms, *, role: str, gradient: bool) -> Atoms:
    item = structure.copy()
    item.calc = None
    item.info["r2o_role"] = role
    item.info["r2o_enters_gradients"] = bool(gradient)
    item.info["r2o_energy_label_used"] = False
    item.info["config_energy_weight"] = 0.0
    item.info["config_forces_weight"] = 1.0
    return item


def edge_audit(
    structures: list[Atoms], references: dict[int, Atoms]
) -> dict:
    by_size: dict[str, dict] = {}
    for size, reference in references.items():
        candidate = undirected_edges(reference, R_MAX_A + EDGE_BUFFER_A)
        active_reference = undirected_edges(reference, R_MAX_A)
        if not active_reference.issubset(candidate):
            raise RuntimeError("reference active edges escaped the buffered graph")
        relevant = [item for item in structures if len(item) == len(reference)]
        missing: list[tuple[int, tuple]] = []
        topology_changes: list[int] = []
        maximum_displacement = 0.0
        maximum_current_active_distance = 0.0
        minimum_current_fourth_shell_distance = float("inf")
        fourth_shell = undirected_edges(reference, 4.0) - active_reference
        for frame, item in enumerate(relevant):
            displacement = minimum_image_displacement(item, reference)
            maximum_displacement = max(
                maximum_displacement,
                float(np.max(np.linalg.norm(displacement, axis=1))),
            )
            active = undirected_edges(item, R_MAX_A)
            if active != active_reference:
                topology_changes.append(frame)
            for key in sorted(active - candidate):
                missing.append((frame, key))
            for key in active:
                maximum_current_active_distance = max(
                    maximum_current_active_distance,
                    edge_length(item, key),
                )
            for key in fourth_shell:
                minimum_current_fourth_shell_distance = min(
                    minimum_current_fourth_shell_distance, edge_length(item, key)
                )
        if missing:
            raise ValueError(
                f"fixed buffered graph misses {len(missing)} active edges for {size}x{size}"
            )
        if topology_changes:
            raise ValueError(
                f"r_max={R_MAX_A:g} graph topology changed in {size}x{size} frames {topology_changes}"
            )
        by_size[f"{size}x{size}"] = {
            "n_structures_audited": len(relevant),
            "reference_active_undirected_edges": len(active_reference),
            "buffered_candidate_undirected_edges": len(candidate),
            "maximum_atom_displacement_A": maximum_displacement,
            "maximum_current_third_shell_distance_A": maximum_current_active_distance,
            "minimum_current_fourth_shell_distance_A": minimum_current_fourth_shell_distance,
            "lower_cutoff_margin_A": R_MAX_A - maximum_current_active_distance,
            "upper_cutoff_margin_A": minimum_current_fourth_shell_distance - R_MAX_A,
            "candidate_cutoff_A": R_MAX_A + EDGE_BUFFER_A,
            "all_current_active_edges_covered": True,
            "all_current_edge_sets_identical_to_reference": True,
        }
    return by_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs = {name: args.source_root / name for name in EXPECTED_INPUT_HASHES}
    observed = {name: sha256(path) for name, path in inputs.items()}
    if observed != EXPECTED_INPUT_HASHES:
        raise ValueError(
            f"R2O frozen source mismatch: observed={observed}, expected={EXPECTED_INPUT_HASHES}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output {args.output_dir}")

    source_manifest = json.loads(inputs["manifest.json"].read_text(encoding="utf-8"))
    if source_manifest.get("status") != "frozen_before_R2M_support_free_core_training":
        raise ValueError("wrong R2M source status")
    if source_manifest.get("model_role") != "replacement_short_delta_core":
        raise ValueError("wrong R2M target semantics")

    train_source = read(inputs["train.xyz"], index=":")
    valid_source = read(inputs["valid.xyz"], index=":")
    if (len(train_source), len(valid_source)) != (164, 45):
        raise ValueError("unexpected R2M source counts")

    train_thermal: list[Atoms] = []
    harmonic_train_source: list[Atoms] = []
    for item in train_source:
        role = str(item.info.get("r2m_target_role", ""))
        if role == "exact_e50_seed0":
            if int(item.info.get("trajectory_seed", -1)) != 0:
                raise ValueError("non-seed0 E50 entered R2O gradients")
            copied = clone(item, role="exact_e50_seed0_train", gradient=True)
            copied.info["config_type"] = "r2o_exact_e50_seed0_train"
            train_thermal.append(copied)
        elif role == "auxiliary_legacy_thermal":
            temperature = float(item.info.get("lattice_temperature_K", -1.0))
            if temperature not in {300.0, 600.0}:
                raise ValueError("R2O auxiliary temperature changed")
            copied = clone(
                item, role=f"auxiliary_T{int(temperature)}_train", gradient=True
            )
            copied.info["config_type"] = f"r2o_auxiliary_T{int(temperature)}_train"
            train_thermal.append(copied)
        elif role == "harmonic_train":
            harmonic_train_source.append(item)
        else:
            raise ValueError(f"unexpected R2M train role {role!r}")

    e50_seed1: list[Atoms] = []
    harmonic_full: list[Atoms] = []
    for item in valid_source:
        role = str(item.info.get("r2m_target_role", ""))
        if role == "exact_e50_seed1":
            if int(item.info.get("trajectory_seed", -1)) != 1:
                raise ValueError("wrong E50 endpoint seed")
            copied = clone(item, role="exact_e50_seed1_gate", gradient=False)
            copied.info["config_type"] = "r2o_exact_e50_seed1_gate"
            e50_seed1.append(copied)
        elif role == "harmonic_validation":
            copied = clone(item, role="harmonic_full_report", gradient=False)
            copied.info["config_type"] = "r2o_harmonic_full_report"
            harmonic_full.append(copied)
        else:
            raise ValueError(f"unexpected R2M validation role {role!r}")

    train_counts = Counter(item.info["r2o_role"] for item in train_thermal)
    if train_counts != Counter(
        {
            "exact_e50_seed0_train": 20,
            "auxiliary_T300_train": 36,
            "auxiliary_T600_train": 36,
        }
    ):
        raise ValueError(f"wrong R2O thermal-only counts: {train_counts}")
    if (len(e50_seed1), len(harmonic_full)) != (20, 25):
        raise ValueError("wrong R2O endpoint/report counts")

    e50_identity_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(item.arrays["REF_forces"], float)
                    - (
                        np.asarray(item.arrays["DFT_TOTAL_forces"], float)
                        - np.asarray(item.arrays["FOUNDATION_BASE_forces"], float)
                        - np.asarray(item.arrays["FROZEN_Q6_forces"], float)
                    )
                )
            )
        )
        for item in train_thermal + e50_seed1
        if str(item.info["r2o_role"]).startswith("exact_e50")
    )
    if e50_identity_error > 1.1e-8:
        raise ValueError("E50 target is not DFT_TOTAL-FOUNDATION_BASE-FROZEN_Q6")
    auxiliary = [
        item for item in train_thermal if str(item.info["r2o_role"]).startswith("auxiliary")
    ]
    auxiliary_identity_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(item.arrays["REF_forces"], float)
                    - (
                        np.asarray(item.arrays["TOTAL_forces"], float)
                        - np.asarray(item.arrays["BASE_forces"], float)
                        - np.asarray(item.arrays["LONG_RANGE_forces"], float)
                    )
                )
            )
        )
        for item in auxiliary
    )
    if auxiliary_identity_error > 2.0e-15:
        raise ValueError("auxiliary target is not TOTAL-BASE-LONG_RANGE")
    auxiliary_degauss = {
        int(float(item.info["lattice_temperature_K"])): float(item.info["degauss_Ry"])
        for item in auxiliary
    }
    if auxiliary_degauss != {
        300: FIXED_DEGAUSS_RY,
        600: MATCHED_T600_DEGAUSS_RY,
    }:
        raise ValueError(f"auxiliary matched-smearing metadata changed: {auxiliary_degauss}")

    reference6, size6 = graphene_reference(train_thermal[0])
    reference8, size8 = graphene_reference(harmonic_full[0])
    if (size6, size8) != (6, 8):
        raise ValueError("R2O reference sizes changed")
    references = {6: reference6, 8: reference8}
    for item in train_thermal + e50_seed1:
        reference, size = graphene_reference(item)
        if size != 6 or not np.allclose(reference.positions, reference6.positions):
            raise ValueError("6x6 reference/order mismatch")
    for item in harmonic_full:
        reference, size = graphene_reference(item)
        if size != 8 or not np.allclose(reference.positions, reference8.positions):
            raise ValueError("8x8 reference/order mismatch")

    assignment_records = {"6x6": [], "8x8": []}
    geometry_audit_pool = train_source + valid_source
    for item in geometry_audit_pool:
        size = 6 if len(item) == 72 else 8
        reference = references[size]
        mapping, record = reference_assignment(item, reference)
        if not record["identity"] or not record["unique_nearest"]:
            raise ValueError("source-to-phonopy mapping is not uniquely identity")
        nearest_bond = min(
            edge_length(reference, key) for key in nearest_bond_keys(reference)
        )
        if record["maximum_assigned_distance_A"] >= 0.5 * nearest_bond:
            raise ValueError("source/reference displacement exceeds half a C-C bond")
        if not np.array_equal(mapping, np.arange(len(item))):
            raise RuntimeError("identity mapping assertion changed")
        assignment_records[f"{size}x{size}"].append(record)
    assignment_report = {
        key: {
            "n_structures": len(records),
            "all_Hungarian_assignments_identity": True,
            "all_assignments_unique_nearest": True,
            "maximum_assigned_distance_A": max(
                record["maximum_assigned_distance_A"] for record in records
            ),
            "minimum_nearest_second_nearest_gap_A": min(
                record["minimum_nearest_second_nearest_gap_A"] for record in records
            ),
            "maximum_allowed_assigned_distance_A": 0.5
            * min(
                edge_length(references[int(key[0])], bond)
                for bond in nearest_bond_keys(references[int(key[0])])
            ),
        }
        for key, records in assignment_records.items()
    }

    bonds8 = nearest_bond_keys(reference8)
    harmonic_train_small_zero: list[Atoms] = []
    harmonic_train_bond_rms_values = []
    for index, source in enumerate(harmonic_train_source):
        value = bond_rms(source, reference8, bonds8)
        harmonic_train_bond_rms_values.append(value)
        if value <= SMALL_BOND_RMS_MAX_A:
            selected = clone(
                source, role="harmonic_lambda1_small_zero_train", gradient=True
            )
            selected.info["config_type"] = "r2o_harmonic_lambda1_small_zero_train"
            selected.info["r2o_harmonic_train_source_index"] = index
            selected.info["r2o_bond_length_RMS_A"] = value
            selected.arrays["ORIGINAL_SHORT_DELTA_REF_forces"] = np.asarray(
                source.arrays["REF_forces"], float
            ).copy()
            selected.arrays["REF_forces"] = np.zeros((len(selected), 3), float)
            selected.arrays["CORE_TARGET_forces"] = np.zeros((len(selected), 3), float)
            harmonic_train_small_zero.append(selected)
    if not harmonic_train_small_zero:
        raise ValueError("fixed lambda=1 harmonic-train rule selected no frames")
    harmonic_small: list[Atoms] = []
    bond_rms_values = []
    for index, item in enumerate(harmonic_full):
        value = bond_rms(item, reference8, bonds8)
        item.info["r2o_harmonic_index"] = index
        item.info["r2o_bond_length_RMS_A"] = value
        item.info["r2o_small_amplitude_gate"] = value <= SMALL_BOND_RMS_MAX_A
        bond_rms_values.append(value)
        if value <= SMALL_BOND_RMS_MAX_A:
            selected = item.copy()
            selected.info["r2o_role"] = "harmonic_lambda1_small_gate"
            selected.info["config_type"] = "r2o_harmonic_lambda1_small_gate"
            harmonic_small.append(selected)
    if not harmonic_small:
        raise ValueError("fixed 0.003 A bond-RMS gate selected no harmonic frames")
    harmonic_small_zero_tail = force_metrics(harmonic_small)
    harmonic_full_zero_tail = force_metrics(harmonic_full)
    harmonic_small_maximum_MIC_atom_displacement_A = max(
        reference_assignment(item, reference8)[1]["maximum_assigned_distance_A"]
        for item in harmonic_small
    )
    if harmonic_small_maximum_MIC_atom_displacement_A > 0.03 + 1.0e-8:
        raise ValueError("small harmonic gate exceeds the fixed 0.03 A atom-displacement bound")

    gradient_fingerprints = {geometry_fingerprint(item) for item in train_thermal}
    stage2_harmonic_fingerprints = {
        geometry_fingerprint(item) for item in harmonic_train_small_zero
    }
    seed1_fingerprints = {geometry_fingerprint(item) for item in e50_seed1}
    harmonic_fingerprints = {geometry_fingerprint(item) for item in harmonic_full}
    if len(gradient_fingerprints) != 92:
        raise ValueError("R2O gradient geometries are not unique")
    if len(stage2_harmonic_fingerprints) != len(harmonic_train_small_zero):
        raise ValueError("R2O small harmonic train geometries are not unique")
    if gradient_fingerprints & seed1_fingerprints:
        raise ValueError("E50 seed1 geometry leaked into gradients")
    if (gradient_fingerprints | seed1_fingerprints) & harmonic_fingerprints:
        raise ValueError("thermal/harmonic geometry overlap")
    if stage2_harmonic_fingerprints & (
        gradient_fingerprints | seed1_fingerprints | harmonic_fingerprints
    ):
        raise ValueError("stage2 harmonic-zero geometry leaked across a frozen split")

    edge_report = edge_audit(geometry_audit_pool, references)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "train_thermal.xyz": train_thermal,
        "train_harmonic_lambda1_small_zero.xyz": harmonic_train_small_zero,
        "valid_e50_seed1.xyz": e50_seed1,
        "harmonic_lambda1_small_gate.xyz": harmonic_small,
        "harmonic_full_report.xyz": harmonic_full,
        "reference_6x6.xyz": [reference6],
        "reference_8x8.xyz": [reference8],
    }
    for name, structures in outputs.items():
        atomic_extxyz(args.output_dir / name, structures)
    shutil.copy2(__file__, args.output_dir / "prepare_script_snapshot.py")

    output_records = {
        name: {"path": str(args.output_dir / name), "sha256": sha256(args.output_dir / name)}
        for name in outputs
    }
    output_records["prepare_script_snapshot.py"] = {
        "path": str(args.output_dir / "prepare_script_snapshot.py"),
        "sha256": sha256(args.output_dir / "prepare_script_snapshot.py"),
    }
    payload = {
        "format": FORMAT,
        "status": "frozen_before_R2O_training",
        "model_role": "anharmonic_short_delta_energy_Taylor_remainder_order3plus",
        "combination": "frozen_v11_foundation + R2O_Taylor_null_core + frozen_q6",
        "inputs": {
            name: {"path": str(path), "sha256": observed[name]}
            for name, path in inputs.items()
        },
        "counts": {
            "gradient_train_thermal_only": len(train_thermal),
            "stage2_gradient_harmonic_lambda1_small_zero": len(
                harmonic_train_small_zero
            ),
            "train_exact_e50_seed0": train_counts["exact_e50_seed0_train"],
            "train_auxiliary_T300": train_counts["auxiliary_T300_train"],
            "train_auxiliary_T600": train_counts["auxiliary_T600_train"],
            "valid_e50_seed1_gate_only": len(e50_seed1),
            "harmonic_lambda1_small_gate": len(harmonic_small),
            "harmonic_full_report_only": len(harmonic_full),
            "seed2_opened": 0,
            "support_opened": 0,
        },
        "leakage_control": {
            "E50_seed1_in_gradients_or_scales": False,
            "E50_seed2_path_is_an_input": False,
            "E50_seed2_file_opened": False,
            "support_path_is_an_input": False,
            "support_file_opened": False,
            "harmonic_frames_in_encoder_pretrain_gradients": False,
            "harmonic_frames_in_stage2_gradients": True,
            "stage2_harmonic_gradient_scope": f"{len(harmonic_train_small_zero)} harmonic-train frames selected only by fixed bond_RMS<=0.003 A and assigned zero tail-force targets",
            "atom_or_bond_random_split": False,
            "gradient_endpoint_geometry_overlap": 0,
        },
        "training_data_contract": {
            "force_only": True,
            "energy_labels_used": False,
            "encoder_pretrain_thermal_only": True,
            "stage2_thermal_plus_small_harmonic_zero": True,
            "fixed_group_mass": {"E50_seed0": 0.5, "T300": 0.25, "T600": 0.25},
            "fixed_gate_scale_meV_A": 30.0,
            "normalized_config_type_weights": {
                "r2o_exact_e50_seed0_train": 1.0,
                "r2o_auxiliary_T300_train": 0.2777777777777778,
                "r2o_auxiliary_T600_train": 0.2777777777777778,
            },
            "stage2_group_mass": {
                "E50_seed0": 0.50,
                "T300": 0.20,
                "T600": 0.20,
                "harmonic_lambda1_small_zero": 0.10,
            },
            "stage2_group_force_scale_meV_A": {
                "E50_seed0": 30.0,
                "T300": 30.0,
                "T600": 30.0,
                "harmonic_lambda1_small_zero": 0.5,
            },
            "stage2_harmonic_target": "R2O Taylor-tail force exactly zero; original short-delta REF retained in ORIGINAL_SHORT_DELTA_REF_forces",
            "target_provenance": {
                "E50_exact_identity": "REF_forces = DFT_TOTAL_forces - FOUNDATION_BASE_forces - FROZEN_Q6_forces",
                "E50_identity_max_abs_error_eV_A": e50_identity_error,
                "E50_identity_tolerance_eV_A": 1.1e-8,
                "auxiliary_identity": "REF_forces = TOTAL_forces - BASE_forces - LONG_RANGE_forces",
                "auxiliary_identity_max_abs_error_eV_A": auxiliary_identity_error,
                "auxiliary_identity_tolerance_eV_A": 2.0e-15,
                "auxiliary_operator_scope": {
                    "T300_degauss_Ry": FIXED_DEGAUSS_RY,
                    "T600_degauss_Ry": MATCHED_T600_DEGAUSS_RY,
                    "description": "matched-smearing local-Mermin transferable delta; T600 is not a fixed-smearing DFT label",
                },
                "all92_are_fixed_smearing_DFT_labels": False,
            },
        },
        "reference_contract": {
            "construction": "td_common.build_monolayer(graphene,vacuum=7.5,a=2.4600000087) then Phonopy diag(n,n,1), primitive_matrix=I, phonopy_to_ase and wrap; fractional coordinates projected onto the audited source cell",
            "minimum_image_displacement": True,
            "minimum_image_integer_shift_frozen_outside_autograd": True,
            "allowed_supercells": [6, 8],
            "reference_6x6_fingerprint": geometry_fingerprint(reference6),
            "reference_8x8_fingerprint": geometry_fingerprint(reference8),
            "source_to_reference_assignment": assignment_report,
            "runtime_policy": "solve a species-aware MIC Hungarian assignment, reject non-unique mappings, reorder to reference order, and map predicted forces back to source order",
        },
        "edge_contract": {
            "model_r_max_A": R_MAX_A,
            "fixed_reference_edge_buffer_A": EDGE_BUFFER_A,
            "runtime_must_reuse_buffered_reference_edges": True,
            "runtime_must_reject_uncovered_active_edges": True,
            "audit": edge_report,
        },
        "harmonic_gate_contract": {
            "selection": "lambda=1 small-amplitude frames only",
            "bond_length_RMS_definition": "RMS current-minus-reference length over the 3N/2 pristine nearest-neighbour bonds with periodic images fixed by reference",
            "bond_length_RMS_max_A": SMALL_BOND_RMS_MAX_A,
            "gate_kind": "internal validation threshold fixed before training and checkpoint selection; not an external blind test",
            "small12_maximum_MIC_atom_displacement_A": harmonic_small_maximum_MIC_atom_displacement_A,
            "small12_maximum_MIC_atom_displacement_limit_A": 0.03,
            "small12_atom_displacement_numerical_tolerance_A": 1.0e-8,
            "selected_source_indices": [
                int(item.info["r2o_harmonic_index"]) for item in harmonic_small
            ],
            "all25_bond_length_RMS_A": bond_rms_values,
            "stage2_small_zero_train_source_indices": [
                int(item.info["r2o_harmonic_train_source_index"])
                for item in harmonic_train_small_zero
            ],
            "all72_harmonic_train_bond_length_RMS_A": harmonic_train_bond_rms_values,
            "stage2_small_zero_group_mass": 0.10,
            "zero_tail_small12_baseline": harmonic_small_zero_tail,
            "small_gate_fixed_RMSE_meV_A": 0.5,
            "small_gate_fixed_max_abs_meV_A": 10.0,
            "small_gate_margin_over_zero_tail": {
                "RMSE_ratio": 0.5 / harmonic_small_zero_tail["RMSE_meV_A"],
                "max_abs_ratio": 10.0
                / harmonic_small_zero_tail["max_abs_meV_A"],
            },
            "full25_policy": "linear pseudo-replay is report-only and cannot select a checkpoint",
            "full25_zero_tail_baseline_report": harmonic_full_zero_tail,
            "exact_reference_Hessian_required": True,
            "Gamma_K_frequency_drift_required": True,
        },
        "fixed_endpoint_thresholds": {
            "e50_seed1_force_RMSE_meV_A": 30.0,
            "e50_seed1_force_max_abs_meV_A": 200.0,
            "e50_seed1_Aprime_RMS_meV_A": 15.0,
            "e50_seed1_Aprime_slope_relative_error_abs": 0.05,
            "harmonic_small_force_RMSE_meV_A": 0.5,
            "harmonic_small_force_max_abs_meV_A": 10.0,
            "harmonic_full25_old_RMSE_meV_A_report_only": 9.044673057081592,
            "harmonic_full25_old_max_abs_meV_A_report_only": 200.0,
            "Taylor_remainder_reference_E_abs_eV": 1.0e-10,
            "Taylor_remainder_reference_force_max_abs_eV_A": 1.0e-9,
            "Taylor_remainder_reference_Hessian_max_abs_eV_A2": 1.0e-7,
            "Taylor_remainder_reference_Hessian_ASR_row_sum_max_abs_eV_A2": 1.0e-7,
            "Gamma_K_frequency_drift_cm-1": 2.0,
        },
        "architecture_freeze": {
            "model": "MACE energy Taylor remainder",
            "r_max_A": R_MAX_A,
            "num_interactions": 2,
            "hidden_irreps": "16x0e+16x1o+16x2e",
            "max_ell": 2,
            "num_radial_basis": 24,
            "num_cutoff_basis": 5,
            "correlation": 3,
            "default_dtype": "float64",
            "dtype_reason": "whole-energy Taylor cancellation and the fixed 1e-10/1e-9 reference-null gates",
            "cutoff_polynomial_order_p": 5,
            "pair_repulsion": False,
            "pair_repulsion_policy": "formal architecture validation fails closed if a pair-repulsion module is present",
            "cutoff_regularitiy": "C2: cutoff value, first derivative, and second derivative vanish",
            "interaction_diameter_bound_A": 12.8,
            "minimum_6x6_in_plane_cell_length_A": 14.76,
            "no_periodic_self_wrap_by_diameter_bound": True,
            "encoder_pretraining_epochs": 80,
            "Taylor_remainder_training_epochs": 240,
            "Taylor_order_removed": 2,
            "leading_energy_order_at_reference": 3,
            "feature_jet_abandoned": True,
            "reason": "whole-energy Taylor remainder frozen after the Cartesian live-HVP feasibility audit",
        },
        "outputs": output_records,
    }
    strict_json(args.output_dir / "manifest.json", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
