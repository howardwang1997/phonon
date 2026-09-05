#!/usr/bin/env python3
"""Select and freeze a 12-structure DFT overlap screen from an SSCHA ensemble."""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from mace.calculators import MACECalculator
from scipy.optimize import linear_sum_assignment

from audit_graphene_current_s0_fixed_smearing import (
    atomic_json,
    atomic_npz,
    folded_k_aprime_mode,
    load_operator,
    minimum_image_vectors,
    sha256,
)


def torch_load(path: Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"_epoch-(\d+)(?:_swa)?\.pt$", path.name)
    if not match:
        raise ValueError(f"cannot parse checkpoint epoch from {path}")
    return int(match.group(1))


def fixed_mapping(
    positions: np.ndarray, reference: np.ndarray, cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mappings = []
    displacements = []
    for frame in positions:
        vectors = minimum_image_vectors(frame, reference, cell)
        row, column = linear_sum_assignment(np.linalg.norm(vectors, axis=2))
        if not np.array_equal(row, np.arange(len(frame))):
            column = column[np.argsort(row)]
        mappings.append(column.astype(int))
        displacements.append(vectors[np.arange(len(frame)), column])
    mappings = np.asarray(mappings)
    unique = {tuple(value.tolist()) for value in mappings}
    if len(unique) != 1:
        raise ValueError(f"SSCHA ensemble has {len(unique)} distinct atom mappings")
    return mappings[0], np.asarray(displacements)


def write_extxyz_atomic(path: Path, structures: list[Atoms]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write(temporary, structures, format="extxyz")
    os.replace(temporary, path)


def standardized_features(raw: np.ndarray):
    median = np.median(raw, axis=0)
    lower = np.percentile(raw, 25.0, axis=0)
    upper = np.percentile(raw, 75.0, axis=0)
    scale = upper - lower
    standard = np.std(raw, axis=0)
    scale = np.where(scale > 1.0e-12, scale, standard)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (raw - median) / scale, median, scale


def diverse_selection(
    pool: np.ndarray,
    count: int,
    features: np.ndarray,
    already: list[int],
    priority: np.ndarray | None = None,
    first_closest_to_origin: bool = False,
) -> list[int]:
    available = [int(value) for value in pool if int(value) not in already]
    chosen = []
    if not available:
        return chosen
    while len(chosen) < count and available:
        existing = already + chosen
        candidate_features = features[available]
        if not existing:
            if first_closest_to_origin and not chosen:
                pick_local = int(np.argmin(np.linalg.norm(candidate_features, axis=1)))
            elif priority is not None:
                pick_local = int(np.argmax(priority[available]))
            else:
                pick_local = int(np.argmax(np.linalg.norm(candidate_features, axis=1)))
        else:
            distances = np.linalg.norm(
                candidate_features[:, None, :] - features[np.asarray(existing)][None, :, :],
                axis=2,
            ).min(axis=1)
            score = distances
            if priority is not None:
                values = priority[available]
                spread = float(np.ptp(values))
                normalized = (
                    (values - float(np.min(values))) / spread
                    if spread > 1.0e-14
                    else np.zeros_like(values)
                )
                score = score + 0.35 * normalized
            pick_local = int(np.argmax(score))
        chosen.append(available.pop(pick_local))
    return chosen


def materialize_structures(
    positions: np.ndarray,
    forces: np.ndarray,
    energies: np.ndarray,
    cell: np.ndarray,
    source_indices: np.ndarray,
    groups: dict[int, str],
    degauss: float,
) -> list[Atoms]:
    structures = []
    for local_index, source_index in enumerate(source_indices):
        atoms = Atoms(
            numbers=np.full(positions.shape[1], 6, int),
            positions=positions[source_index],
            cell=cell,
            pbc=(True, True, False),
        )
        atoms.wrap(eps=1.0e-12)
        atoms.info.update(
            config_type=f"s0_sscha_overlap_{int(source_index):03d}",
            split="frozen_overlap_screen",
            sscha_index=int(source_index),
            overlap_local_index=int(local_index),
            selection_group=groups[int(source_index)],
            lattice_temperature_K=450.0,
            degauss_Ry=degauss,
            sscha_random_seed=453005,
            MLIP_energy=float(energies[source_index]),
        )
        atoms.arrays["MLIP_forces"] = np.asarray(forces[source_index], float)
        structures.append(atoms)
    return structures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xats", type=Path, required=True)
    parser.add_argument("--forces", type=Path, required=True)
    parser.add_argument("--energies", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--thermal-result", type=Path, required=True)
    parser.add_argument("--template-model", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--committee-epochs", default="195,205,215")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-select", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.n_select != 12:
        raise ValueError("the frozen R1 protocol requires exactly 12 structures")

    xats = np.asarray(np.load(args.xats), float)
    saved_forces = np.asarray(np.load(args.forces), float)
    saved_energies = np.asarray(np.load(args.energies), float)
    if xats.shape != (300, 72, 3) or saved_forces.shape != xats.shape:
        raise ValueError("expected 300 SSCHA structures with 72 atoms")
    if saved_energies.shape != (300,):
        raise ValueError("expected 300 SSCHA energies")
    operator_fc, reference, cell, degauss = load_operator(args.operator)
    del operator_fc
    mapping, displacement_xats_order = fixed_mapping(xats, reference, cell)
    reorder = np.argsort(mapping)
    positions = xats[:, reorder]
    saved_forces = saved_forces[:, reorder]
    displacement = displacement_xats_order[:, reorder]
    if float(np.max(np.linalg.norm(displacement, axis=2))) > 0.8:
        raise ValueError("SSCHA ensemble contains a displacement beyond 0.8 angstrom")

    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.thermal_result, reference, cell
    )
    mode_flat = mode.reshape(-1)
    aprime_coordinate = np.asarray(
        [np.vdot(mode_flat, value.reshape(-1)) for value in displacement]
    )
    aprime_force = np.asarray(
        [np.vdot(mode_flat, value.reshape(-1)) for value in saved_forces]
    )

    requested_epochs = [int(value) for value in args.committee_epochs.split(",")]
    if len(requested_epochs) < 3 or len(set(requested_epochs)) != len(requested_epochs):
        raise ValueError("committee requires at least three distinct epochs")
    by_epoch = {
        checkpoint_epoch(path): path
        for path in args.checkpoint_dir.glob("*_epoch-*.pt")
        if "_swa" not in path.name
    }
    missing = [epoch for epoch in requested_epochs if epoch not in by_epoch]
    if missing:
        raise ValueError(f"missing committee checkpoints: {missing}")
    template = torch_load(args.template_model, "cpu")
    structures_for_inference = [
        Atoms(
            numbers=np.full(72, 6, int),
            positions=value,
            cell=cell,
            pbc=(True, True, False),
        )
        for value in positions
    ]
    committee_forces = []
    committee_records = []
    for epoch in requested_epochs:
        checkpoint_path = by_epoch[epoch]
        checkpoint = torch_load(checkpoint_path, "cpu")
        candidate = copy.deepcopy(template)
        candidate.load_state_dict(checkpoint["model"], strict=True)
        calculator = MACECalculator(
            models=candidate, device=args.device, default_dtype="float32"
        )
        values = []
        for structure in structures_for_inference:
            atoms = structure.copy()
            atoms.calc = calculator
            values.append(np.asarray(atoms.get_forces(), float))
        committee_forces.append(values)
        committee_records.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
            }
        )
        del calculator, candidate, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    committee_forces = np.asarray(committee_forces)
    force_disagreement = np.sqrt(np.mean(np.var(committee_forces, axis=0), axis=(1, 2)))

    rms_displacement = np.sqrt(np.mean(displacement**2, axis=(1, 2)))
    max_displacement = np.max(np.linalg.norm(displacement, axis=2), axis=1)
    inplane_rms = np.sqrt(np.mean(displacement[:, :, :2] ** 2, axis=(1, 2)))
    outplane_rms = np.sqrt(np.mean(displacement[:, :, 2] ** 2, axis=1))
    force_rms = np.sqrt(np.mean(saved_forces**2, axis=(1, 2)))
    energy_centered = saved_energies - np.median(saved_energies)
    raw_features = np.column_stack(
        [
            rms_displacement,
            max_displacement,
            inplane_rms,
            outplane_rms,
            np.abs(aprime_coordinate),
            force_rms,
            force_disagreement,
            energy_centered,
        ]
    )
    feature_names = [
        "RMS_displacement_A",
        "max_displacement_A",
        "inplane_RMS_displacement_A",
        "outplane_RMS_displacement_A",
        "Aprime_coordinate_abs_A",
        "MLIP_force_RMS_eV_A",
        "committee_force_disagreement_RMS_eV_A",
        "MLIP_energy_centered_eV",
    ]
    features, feature_median, feature_scale = standardized_features(raw_features)

    amplitude = np.abs(aprime_coordinate)
    typical_pool = np.flatnonzero(
        (amplitude >= np.percentile(amplitude, 25.0))
        & (amplitude <= np.percentile(amplitude, 75.0))
        & (force_disagreement <= np.percentile(force_disagreement, 60.0))
        & (rms_displacement >= np.percentile(rms_displacement, 10.0))
        & (rms_displacement <= np.percentile(rms_displacement, 90.0))
    )
    high_disagreement_pool = np.flatnonzero(
        force_disagreement >= np.percentile(force_disagreement, 80.0)
    )
    low_aprime_pool = np.flatnonzero(amplitude <= np.percentile(amplitude, 10.0))
    high_aprime_pool = np.flatnonzero(amplitude >= np.percentile(amplitude, 90.0))

    selected: list[int] = []
    groups: dict[int, str] = {}
    typical = diverse_selection(
        typical_pool, 4, features, selected, first_closest_to_origin=True
    )
    selected.extend(typical)
    groups.update({index: "typical" for index in typical})
    disagreement = diverse_selection(
        high_disagreement_pool,
        4,
        features,
        selected,
        priority=force_disagreement,
    )
    selected.extend(disagreement)
    groups.update({index: "high_committee_disagreement" for index in disagreement})
    low = diverse_selection(low_aprime_pool, 2, features, selected)
    selected.extend(low)
    groups.update({index: "low_Aprime_amplitude" for index in low})
    high = diverse_selection(
        high_aprime_pool, 2, features, selected, priority=amplitude
    )
    selected.extend(high)
    groups.update({index: "high_Aprime_amplitude" for index in high})
    if len(selected) != 12 or len(set(selected)) != 12:
        raise ValueError(f"selection produced {len(selected)} non-unique structures")
    selected_array = np.asarray(selected, int)

    difficulty = (
        features[:, feature_names.index("MLIP_force_RMS_eV_A")]
        + features[:, feature_names.index("max_displacement_A")]
        + features[:, feature_names.index("committee_force_disagreement_RMS_eV_A")]
    )
    shard_members = {"A": [], "B": []}
    shard_scores = {"A": 0.0, "B": 0.0}
    for index in sorted(selected, key=lambda value: float(difficulty[value]), reverse=True):
        choices = [name for name in ("A", "B") if len(shard_members[name]) < 6]
        shard = min(choices, key=lambda name: shard_scores[name])
        shard_members[shard].append(index)
        shard_scores[shard] += float(difficulty[index])
    if any(len(values) != 6 for values in shard_members.values()):
        raise ValueError("failed to balance the two six-structure shards")

    structures = materialize_structures(
        positions,
        saved_forces,
        saved_energies,
        cell,
        selected_array,
        groups,
        degauss,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_xyz = args.output_dir / "selected12.extxyz"
    write_extxyz_atomic(selected_xyz, structures)
    all_selected_npz = args.output_dir / "selected12_snapshots.npz"
    atomic_npz(
        all_selected_npz,
        positions=positions[selected_array],
        cells=np.repeat(cell[None, :, :], 12, axis=0),
        numbers=np.full(72, 6, int),
        sscha_indices=selected_array,
        selection_groups=np.asarray([groups[int(value)] for value in selected_array]),
        MLIP_forces=saved_forces[selected_array],
        MLIP_energies=saved_energies[selected_array],
    )

    for shard, members in shard_members.items():
        indices = np.asarray(members, int)
        shard_path = args.output_dir / f"shard_{shard}_snapshots.npz"
        atomic_npz(
            shard_path,
            positions=positions[indices],
            cells=np.repeat(cell[None, :, :], len(indices), axis=0),
            numbers=np.full(72, 6, int),
            sscha_indices=indices,
            selection_groups=np.asarray([groups[int(value)] for value in indices]),
            MLIP_forces=saved_forces[indices],
            MLIP_energies=saved_energies[indices],
        )
        shard_structures = materialize_structures(
            positions,
            saved_forces,
            saved_energies,
            cell,
            indices,
            groups,
            degauss,
        )
        write_extxyz_atomic(args.output_dir / f"shard_{shard}.extxyz", shard_structures)

    rows = []
    for index in range(len(xats)):
        row = {
            "sscha_index": index,
            "selected": index in groups,
            "selection_group": groups.get(index, ""),
            "shard": next(
                (name for name, values in shard_members.items() if index in values), ""
            ),
        }
        row.update(
            {name: float(value) for name, value in zip(feature_names, raw_features[index])}
        )
        row["Aprime_force_abs_eV_A"] = float(abs(aprime_force[index]))
        rows.append(row)
    csv_path = args.output_dir / "candidate_features.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_npz(
        args.output_dir / "selection_diagnostics.npz",
        raw_features=raw_features,
        standardized_features=features,
        feature_names=np.asarray(feature_names),
        feature_median=feature_median,
        feature_scale=feature_scale,
        committee_delta_forces_eV_A=committee_forces,
        committee_force_disagreement_RMS_eV_A=force_disagreement,
        displacement_A=displacement,
        Aprime_mode_operator_order=mode,
        Aprime_coordinate_A=aprime_coordinate,
        Aprime_saved_force_eV_A=aprime_force,
        sscha_to_operator_mapping=mapping,
        operator_reorder_indices=reorder,
        selected_indices=selected_array,
    )

    selected_records = []
    for index in selected:
        selected_records.append(
            {
                "sscha_index": index,
                "selection_group": groups[index],
                "shard": next(
                    name for name, values in shard_members.items() if index in values
                ),
                "features": {
                    name: float(value)
                    for name, value in zip(feature_names, raw_features[index])
                },
            }
        )
    manifest = {
        "status": "frozen_before_R1_DFT",
        "scope": "current S0 self-consistent 450 K lattice ensemble at fixed electronic smearing",
        "selection_protocol": {
            "n_candidates": 300,
            "n_selected": 12,
            "groups": {
                "typical": 4,
                "high_committee_disagreement": 4,
                "low_Aprime_amplitude": 2,
                "high_Aprime_amplitude": 2,
            },
            "diversity": "robust-standardized physical features with farthest-point selection",
            "committee_epochs": requested_epochs,
            "DFT_labels_read_during_selection": False,
        },
        "condition": {
            "lattice_temperature_K": 450.0,
            "smearing": "fermi-dirac",
            "degauss_Ry": degauss,
            "cell": "fixed operator/SSCHA cell",
            "n_atoms": 72,
        },
        "fixed_DFT_settings": {
            "kgrid": [8, 8, 1],
            "ecutwfc_Ry": 60.0,
            "ecutrho_Ry": 240.0,
            "disk_io": "none",
            "pseudopotential": "C_ONCV_PBE-1.2.upf",
        },
        "fixed_acceptance_thresholds": {
            "force_component_RMSE_meV_A": 30.0,
            "force_component_max_abs_meV_A": 200.0,
            "Aprime_projected_force_RMS_meV_A": 15.0,
            "Aprime_restoring_slope_relative_error": 0.05,
            "centered_energy_RMSE_meV_config": 19.4,
            "importance_weight_ESS_fraction": 0.3,
        },
        "atom_mapping": {
            "SSCHA_xats_row_to_operator_reference": mapping.tolist(),
            "operator_order_source_rows": reorder.tolist(),
            "unique_mapping_across_all_300": True,
            "maximum_displacement_vector_A": float(
                np.max(np.linalg.norm(displacement, axis=2))
            ),
        },
        "Aprime_mode": mode_provenance,
        "committee": committee_records,
        "selected": selected_records,
        "shards": shard_members,
        "inputs": {
            "xats": {"path": str(args.xats), "sha256": sha256(args.xats)},
            "forces": {"path": str(args.forces), "sha256": sha256(args.forces)},
            "energies": {"path": str(args.energies), "sha256": sha256(args.energies)},
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
            "background": {
                "path": str(args.background),
                "sha256": sha256(args.background),
            },
            "thermal_result": {
                "path": str(args.thermal_result),
                "sha256": sha256(args.thermal_result),
            },
            "template_model": {
                "path": str(args.template_model),
                "sha256": sha256(args.template_model),
            },
        },
        "outputs": {
            "selected12_extxyz": {
                "path": str(selected_xyz),
                "sha256": sha256(selected_xyz),
            },
            "selected12_snapshots": {
                "path": str(all_selected_npz),
                "sha256": sha256(all_selected_npz),
            },
            "shard_A_snapshots_sha256": sha256(
                args.output_dir / "shard_A_snapshots.npz"
            ),
            "shard_B_snapshots_sha256": sha256(
                args.output_dir / "shard_B_snapshots.npz"
            ),
            "candidate_features_sha256": sha256(csv_path),
        },
        "decision_after_DFT": (
            "evaluate the frozen force/energy gates before any checkpoint, architecture, "
            "long-range parameter, or additional DFT-label decision"
        ),
    }
    atomic_json(args.output_dir / "freeze_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected": selected_records,
                "shards": shard_members,
                "output": str(args.output_dir / "freeze_manifest.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
