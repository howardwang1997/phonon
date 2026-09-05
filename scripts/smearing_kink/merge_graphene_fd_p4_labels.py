#!/usr/bin/env python3
"""Merge the fixed P4 force-label shards using ``(trajectory seed, index)`` keys."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from graphene_fd_p4_common import atomic_json, p4_key, sha256  # noqa: E402


COMMON_KEYS = (
    "n_atoms",
    "lattice_temperature_K",
    "smearing",
    "degauss_Ry",
    "ecutwfc_Ry",
    "ecutrho_Ry",
    "disk_io",
    "kgrids",
    "reference_kgrid",
)


def atomic_extxyz(path: Path, structures) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write(temporary, structures, format="extxyz")
    os.replace(temporary, path)


def require_passed_sampling(path: Path, bootstrap: bool) -> dict:
    payload = json.loads(path.read_text())
    passed_key = "passes_sampling_bootstrap_gate" if bootstrap else "passes_sampling_gate"
    if payload.get("status") != "passed" or not payload.get(passed_key):
        raise ValueError(f"sampling prerequisite did not pass: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--sampling-acceptance", type=Path, required=True)
    parser.add_argument("--sampling-bootstrap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze_manifest.read_text())
    sampling = require_passed_sampling(args.sampling_acceptance, bootstrap=False)
    bootstrap = require_passed_sampling(args.sampling_bootstrap, bootstrap=True)
    freeze_sha = sha256(args.freeze_manifest)
    sampling_sha = sha256(args.sampling_acceptance)
    bootstrap_sha = sha256(args.sampling_bootstrap)
    if sampling["freeze_manifest"]["sha256"] != freeze_sha:
        raise ValueError("sampling acceptance refers to a different freeze manifest")
    if bootstrap["freeze_manifest_sha256"] != freeze_sha:
        raise ValueError("sampling bootstrap refers to a different freeze manifest")
    if bootstrap["sampling_acceptance_sha256"] != sampling_sha:
        raise ValueError("sampling bootstrap refers to a different point acceptance")

    specification = freeze["T450_on_policy"]
    expected_indices = [int(value) for value in specification["dft_label_indices_per_seed"]]
    midpoint = len(expected_indices) // 2
    lane_specs = (
        ("A", 0, expected_indices),
        ("A", 2, expected_indices[:midpoint]),
        ("B", 1, expected_indices),
        ("B", 2, expected_indices[midpoint:]),
    )
    shard_manifests = []
    for shard in ("A", "B"):
        marker = args.labels_root / f"shard_{shard}" / "RAW_READY"
        if not marker.is_file():
            raise FileNotFoundError(marker)
        input_manifest_path = args.labels_root / f"shard_{shard}" / "input_manifest.json"
        input_manifest = json.loads(input_manifest_path.read_text())
        if input_manifest.get("status") != "validated_for_dft_force_labels":
            raise ValueError(f"shard {shard} input validation is incomplete")
        if input_manifest.get("shard") != shard:
            raise ValueError(f"shard {shard} input manifest identity mismatch")
        if int(input_manifest.get("n_selected_structures", -1)) != 30:
            raise ValueError(f"shard {shard} input manifest does not contain 30 structures")
        expected_hashes = {
            "freeze_manifest_sha256": freeze_sha,
            "sampling_acceptance_sha256": sampling_sha,
            "sampling_bootstrap_sha256": bootstrap_sha,
        }
        for key, expected_hash in expected_hashes.items():
            if input_manifest.get(key) != expected_hash:
                raise ValueError(f"shard {shard} {key} mismatch")
        shard_manifests.append(
            {
                "shard": shard,
                "path": str(input_manifest_path),
                "sha256": sha256(input_manifest_path),
            }
        )

    observed: dict[tuple[int, int], object] = {}
    records = []
    sources = []
    reference_settings = None
    for shard, seed, expected_lane_indices in lane_specs:
        lane = args.labels_root / f"shard_{shard}" / f"seed{seed}"
        summary_path = lane / "summary.json"
        xyz_path = lane / "summary.xyz"
        payload = json.loads(summary_path.read_text())
        structures = read(xyz_path, index=":")
        payload_indices = [int(value) for value in payload["indices"]]
        if payload_indices != expected_lane_indices:
            raise ValueError(
                f"shard {shard} seed {seed} indices changed: "
                f"{payload_indices} != {expected_lane_indices}"
            )
        if int(payload["trajectory_seed"]) != seed:
            raise ValueError(f"shard {shard} seed metadata mismatch")
        if len(structures) != len(payload_indices):
            raise ValueError(f"shard {shard} seed {seed} JSON/XYZ count mismatch")
        settings = {key: payload[key] for key in COMMON_KEYS}
        if reference_settings is None:
            reference_settings = settings
        elif settings != reference_settings:
            raise ValueError("P4 shard calculation settings differ")

        payload_records = {
            (int(item["trajectory_seed"]), int(item["snapshot_index"])): item
            for item in payload["records"]
        }
        for expected_index, structure in zip(payload_indices, structures, strict=True):
            observed_seed = int(structure.info.get("trajectory_seed", seed))
            observed_index = int(structure.info["snapshot_index"])
            key = (observed_seed, observed_index)
            if key != (seed, expected_index):
                raise ValueError(f"XYZ identity mismatch in shard {shard}: {key}")
            if key in observed:
                raise ValueError(f"duplicate P4 label key: {p4_key(*key)}")
            if str(structure.info.get("split")) != "validation":
                raise ValueError(f"non-validation structure in {p4_key(*key)}")
            if "REF_forces" not in structure.arrays or "REF_energy" not in structure.info:
                raise ValueError(f"missing DFT label in {p4_key(*key)}")
            if key not in payload_records:
                raise ValueError(f"summary record missing for {p4_key(*key)}")
            structure.info["trajectory_seed"] = seed
            structure.info["snapshot_index"] = expected_index
            structure.info["p4_key"] = p4_key(seed, expected_index)
            observed[key] = structure
            records.append(payload_records[key])
        sources.append(
            {
                "shard": shard,
                "trajectory_seed": seed,
                "summary_json": str(summary_path),
                "summary_json_sha256": sha256(summary_path),
                "summary_xyz": str(xyz_path),
                "summary_xyz_sha256": sha256(xyz_path),
                "indices": expected_lane_indices,
            }
        )

    expected_keys = {
        (seed, snapshot_index)
        for seed in (0, 1, 2)
        for snapshot_index in expected_indices
    }
    if set(observed) != expected_keys:
        missing = sorted(expected_keys - set(observed))
        extra = sorted(set(observed) - expected_keys)
        raise ValueError(f"P4 label union mismatch; missing={missing}, extra={extra}")

    output_files = {}
    all_structures = []
    for seed in (0, 1, 2):
        structures = [observed[(seed, index)] for index in expected_indices]
        path = args.output_dir / f"seed{seed}.xyz"
        atomic_extxyz(path, structures)
        output_files[str(seed)] = {
            "path": str(path),
            "sha256": sha256(path),
            "n_structures": len(structures),
            "indices": expected_indices,
        }
        all_structures.extend(structures)
    all_path = args.output_dir / "all60.xyz"
    atomic_extxyz(all_path, all_structures)

    result = {
        "status": "complete",
        "scope": (
            "fixed 450 K on-policy P4 DFT force labels; merge only, with no "
            "training, fitting, or checkpoint selection"
        ),
        "identity_key": "(trajectory_seed, snapshot_index)",
        "temperature_K": int(specification["temperature_K"]),
        "degauss_Ry": float(specification["degauss_Ry"]),
        "n_structures": len(all_structures),
        "n_structures_per_seed": len(expected_indices),
        "indices_per_seed": expected_indices,
        **(reference_settings or {}),
        "freeze_manifest": {"path": str(args.freeze_manifest), "sha256": freeze_sha},
        "sampling_acceptance": {
            "path": str(args.sampling_acceptance),
            "sha256": sampling_sha,
        },
        "sampling_bootstrap": {
            "path": str(args.sampling_bootstrap),
            "sha256": bootstrap_sha,
        },
        "validated_shard_inputs": shard_manifests,
        "sources": sources,
        "records": sorted(
            records,
            key=lambda item: (int(item["trajectory_seed"]), int(item["snapshot_index"])),
        ),
        "seed_outputs": output_files,
        "all_output": {
            "path": str(all_path),
            "sha256": sha256(all_path),
        },
    }
    atomic_json(args.output_dir / "manifest.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
