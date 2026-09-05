#!/usr/bin/env python3
"""Fit the matched-smearing 450 K DFT-TDEP reference from E50 labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from phonopy import Phonopy


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import td_common as tdc  # noqa: E402
import td_phonon as tdp  # noqa: E402
from phonon_accel.phonons import ase_to_phonopy, phonopy_to_ase  # noqa: E402


TARGET_TEMPERATURE_K = 450
TARGET_DEGAUSS_RY = 0.0019000869
EXPECTED_INDICES = (3, 9, 15, 21, 27, 33, 39, 45, 51, 57,
                    63, 69, 75, 81, 87, 93, 99, 105, 111, 117)
CM_PER_THZ = 33.35641


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_extxyz(path: Path, structures) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write(temporary, structures, format="extxyz")
    os.replace(temporary, path)


def evaluated_copy(structure):
    if "REF_forces" not in structure.arrays:
        raise ValueError("DFT label is missing REF_forces")
    result = structure.copy()
    properties = {"forces": np.asarray(structure.arrays["REF_forces"], float)}
    if "REF_energy" in structure.info:
        properties["energy"] = float(structure.info["REF_energy"])
    result.calc = SinglePointCalculator(result, **properties)
    return result


def fit_group(primitive, ideal, sc_matrix, structures, cutoff2: float, npoints: int):
    force_constants, fit_rmse, n_dof = tdp.effective_fc2(
        primitive,
        ideal,
        sc_matrix,
        [evaluated_copy(structure) for structure in structures],
        cutoff2,
    )
    phonon = Phonopy(
        ase_to_phonopy(primitive),
        supercell_matrix=sc_matrix,
        primitive_matrix=np.eye(3),
    )
    distance, frequency, label_positions, labels = tdp.band_from_phonopy(
        phonon, force_constants, npoints=npoints
    )
    k_index = int(np.argmin(np.abs(distance - label_positions[2])))
    return {
        "force_constants": np.asarray(force_constants, float),
        "fit_rmse_eV_A": float(fit_rmse),
        "n_dof": int(n_dof),
        "distance": np.asarray(distance, float),
        "frequency_THz": np.asarray(frequency, float),
        "label_positions": np.asarray(label_positions, float),
        "labels": np.asarray(labels),
        "K_top_cm-1": float(frequency[k_index, -1] * CM_PER_THZ),
        "minimum_frequency_cm-1": float(np.min(frequency) * CM_PER_THZ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=ROOT
        / "results/graphene_physics_temperature/post_p4_feasibility"
        / "E50_fixed_smearing_T450_DFT_reference/labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/graphene_physics_temperature/post_p4_feasibility"
        / "E50_fixed_smearing_T450_DFT_reference",
    )
    parser.add_argument("--a", type=float, default=2.4600000087)
    parser.add_argument("--cutoff2", type=float, default=6.0)
    parser.add_argument("--npoints", type=int, default=201)
    args = parser.parse_args()

    summary_paths = sorted(args.labels_root.glob("shard_*/seed*/summary.json"))
    if len(summary_paths) != 4:
        raise ValueError(f"expected four shard summaries, found {len(summary_paths)}")

    by_key = {}
    provenance = []
    reference_settings = None
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        xyz_path = summary_path.with_suffix(".xyz")
        structures = read(xyz_path, index=":")
        indices = [int(value) for value in payload["indices"]]
        if len(structures) != len(indices):
            raise ValueError(f"JSON/XYZ count mismatch in {summary_path}")
        seed = int(payload["trajectory_seed"])
        settings = {
            "n_atoms": int(payload["n_atoms"]),
            "lattice_temperature_K": float(payload["lattice_temperature_K"]),
            "smearing": str(payload["smearing"]),
            "degauss_Ry": float(payload["degauss_Ry"]),
            "ecutwfc_Ry": float(payload["ecutwfc_Ry"]),
            "ecutrho_Ry": float(payload["ecutrho_Ry"]),
            "reference_kgrid": int(payload["reference_kgrid"]),
        }
        if reference_settings is None:
            reference_settings = settings
        elif settings != reference_settings:
            raise ValueError("DFT calculation settings differ across shards")
        if (
            settings["n_atoms"] != 72
            or settings["lattice_temperature_K"] != TARGET_TEMPERATURE_K
            or settings["smearing"] != "fermi-dirac"
            or abs(settings["degauss_Ry"] - TARGET_DEGAUSS_RY) > 5.0e-11
            or settings["reference_kgrid"] != 8
        ):
            raise ValueError(f"target settings mismatch in {summary_path}")
        for expected_index, structure in zip(indices, structures, strict=True):
            observed_seed = int(structure.info["trajectory_seed"])
            observed_index = int(structure.info["snapshot_index"])
            key = (observed_seed, observed_index)
            if key != (seed, expected_index):
                raise ValueError(f"structure identity mismatch: {key}")
            if key in by_key:
                raise ValueError(f"duplicate DFT label: {key}")
            if abs(float(structure.info["degauss_Ry"]) - TARGET_DEGAUSS_RY) > 5.0e-11:
                raise ValueError(f"structure smearing mismatch: {key}")
            if float(structure.info["lattice_temperature_K"]) != TARGET_TEMPERATURE_K:
                raise ValueError(f"structure temperature mismatch: {key}")
            if not np.isfinite(structure.arrays["REF_forces"]).all():
                raise ValueError(f"non-finite force label: {key}")
            by_key[key] = structure
        provenance.append(
            {
                "summary_json": str(summary_path),
                "summary_json_sha256": sha256(summary_path),
                "summary_xyz": str(xyz_path),
                "summary_xyz_sha256": sha256(xyz_path),
                "trajectory_seed": seed,
                "indices": indices,
            }
        )

    expected_keys = {
        (seed, snapshot_index)
        for seed in (0, 1, 2)
        for snapshot_index in EXPECTED_INDICES
    }
    if set(by_key) != expected_keys:
        raise ValueError(
            f"label union mismatch; missing={sorted(expected_keys-set(by_key))}, "
            f"extra={sorted(set(by_key)-expected_keys)}"
        )

    ordered = [by_key[key] for key in sorted(by_key)]
    by_seed = {
        seed: [by_key[(seed, index)] for index in EXPECTED_INDICES]
        for seed in (0, 1, 2)
    }
    merged_dir = args.output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    for seed in (0, 1, 2):
        atomic_extxyz(merged_dir / f"seed{seed}.xyz", by_seed[seed])
    atomic_extxyz(merged_dir / "all60.xyz", ordered)

    primitive = tdc.build_monolayer("graphene", vacuum=7.5, a=args.a)
    primitive.wrap()
    sc_matrix = np.diag((6, 6, 1))
    geometry_phonon = Phonopy(
        ase_to_phonopy(primitive),
        supercell_matrix=sc_matrix,
        primitive_matrix=np.eye(3),
    )
    ideal = phonopy_to_ase(geometry_phonon.supercell)
    ideal.wrap()
    cell_error = max(
        float(np.max(np.abs(np.asarray(structure.cell) - np.asarray(ideal.cell))))
        for structure in ordered
    )
    if cell_error > 1.0e-5:
        raise ValueError(f"label cell differs from fit cell by {cell_error:g} A")

    pooled = fit_group(
        primitive, ideal, sc_matrix, ordered, args.cutoff2, args.npoints
    )
    seed_fits = {
        seed: fit_group(
            primitive,
            ideal,
            sc_matrix,
            by_seed[seed],
            args.cutoff2,
            args.npoints,
        )
        for seed in (0, 1, 2)
    }
    leave_one_out = {
        omitted: fit_group(
            primitive,
            ideal,
            sc_matrix,
            [
                structure
                for seed in (0, 1, 2)
                if seed != omitted
                for structure in by_seed[seed]
            ],
            args.cutoff2,
            args.npoints,
        )
        for omitted in (0, 1, 2)
    }

    output_npz = args.output_dir / "graphene_fixed_smearing_DFT_TDEP_450K.npz"
    output_json = args.output_dir / "graphene_fixed_smearing_DFT_TDEP_450K.json"
    output_csv = args.output_dir / "graphene_fixed_smearing_DFT_TDEP_450K.csv"
    atomic_npz(
        output_npz,
        temperatures=np.asarray([TARGET_TEMPERATURE_K], float),
        degauss_Ry=np.asarray(TARGET_DEGAUSS_RY),
        n_structures=np.asarray(60),
        fit_rmse_eV_A=np.asarray(pooled["fit_rmse_eV_A"]),
        n_dof=np.asarray(pooled["n_dof"]),
        supercell=np.asarray([6, 6, 1]),
        supercell_matrix=sc_matrix,
        primitive_numbers=np.asarray(primitive.numbers),
        primitive_positions=np.asarray(primitive.positions),
        primitive_cell=np.asarray(primitive.cell),
        label_positions=pooled["label_positions"],
        labels=pooled["labels"],
        T450_dist=pooled["distance"],
        T450_freq=pooled["frequency_THz"],
        T450_fc2=pooled["force_constants"],
        per_seed_frequency_THz=np.asarray(
            [seed_fits[seed]["frequency_THz"] for seed in (0, 1, 2)]
        ),
        per_seed_fc2=np.asarray(
            [seed_fits[seed]["force_constants"] for seed in (0, 1, 2)]
        ),
        per_seed_K_top_cm_1=np.asarray(
            [seed_fits[seed]["K_top_cm-1"] for seed in (0, 1, 2)]
        ),
        leave_one_seed_out_frequency_THz=np.asarray(
            [leave_one_out[seed]["frequency_THz"] for seed in (0, 1, 2)]
        ),
        leave_one_seed_out_fc2=np.asarray(
            [leave_one_out[seed]["force_constants"] for seed in (0, 1, 2)]
        ),
        leave_one_seed_out_K_top_cm_1=np.asarray(
            [leave_one_out[seed]["K_top_cm-1"] for seed in (0, 1, 2)]
        ),
    )

    seed_k = np.asarray(
        [seed_fits[seed]["K_top_cm-1"] for seed in (0, 1, 2)], float
    )
    loo_k = np.asarray(
        [leave_one_out[seed]["K_top_cm-1"] for seed in (0, 1, 2)], float
    )
    summary = {
        "status": "complete",
        "scope": "pooled 60-label fixed-smearing 450 K DFT-TDEP reference",
        "temperature_K": TARGET_TEMPERATURE_K,
        "smearing": "fermi-dirac",
        "degauss_Ry": TARGET_DEGAUSS_RY,
        "n_structures": 60,
        "n_structures_per_seed": 20,
        "fit_RMSE_meV_A": pooled["fit_rmse_eV_A"] * 1000.0,
        "n_fit_parameters": pooled["n_dof"],
        "cutoff2_A": args.cutoff2,
        "K_top_cm-1": pooled["K_top_cm-1"],
        "minimum_frequency_cm-1": pooled["minimum_frequency_cm-1"],
        "per_seed": {
            str(seed): {
                "n_structures": 20,
                "fit_RMSE_meV_A": seed_fits[seed]["fit_rmse_eV_A"] * 1000.0,
                "K_top_cm-1": seed_fits[seed]["K_top_cm-1"],
            }
            for seed in (0, 1, 2)
        },
        "per_seed_K_spread_cm-1": float(np.ptp(seed_k)),
        "leave_one_seed_out": {
            str(seed): {
                "omitted_seed": seed,
                "n_structures": 40,
                "fit_RMSE_meV_A": leave_one_out[seed]["fit_rmse_eV_A"] * 1000.0,
                "K_top_cm-1": leave_one_out[seed]["K_top_cm-1"],
            }
            for seed in (0, 1, 2)
        },
        "leave_one_seed_out_K_spread_cm-1": float(np.ptp(loo_k)),
        "checks": {
            "n_unique_seed_snapshot_keys": len(by_key),
            "cell_max_abs_error_A": cell_error,
            "all_force_labels_finite": True,
            "all_frequencies_finite": bool(
                np.isfinite(pooled["frequency_THz"]).all()
                and all(
                    np.isfinite(fit["frequency_THz"]).all()
                    for fit in [*seed_fits.values(), *leave_one_out.values()]
                )
            ),
        },
        "sources": provenance,
        "outputs": {
            "npz": str(output_npz),
            "csv": str(output_csv),
            "merged_all60": str(merged_dir / "all60.xyz"),
        },
    }
    atomic_json(output_json, summary)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "distance",
                "pooled_top_cm-1",
                "seed0_top_cm-1",
                "seed1_top_cm-1",
                "seed2_top_cm-1",
                "leave_seed0_out_top_cm-1",
                "leave_seed1_out_top_cm-1",
                "leave_seed2_out_top_cm-1",
            ]
        )
        for index, distance in enumerate(pooled["distance"]):
            writer.writerow(
                [
                    distance,
                    pooled["frequency_THz"][index, -1] * CM_PER_THZ,
                    *[
                        seed_fits[seed]["frequency_THz"][index, -1] * CM_PER_THZ
                        for seed in (0, 1, 2)
                    ],
                    *[
                        leave_one_out[seed]["frequency_THz"][index, -1]
                        * CM_PER_THZ
                        for seed in (0, 1, 2)
                    ],
                ]
            )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
