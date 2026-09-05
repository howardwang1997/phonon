#!/usr/bin/env python3
"""Merge disjoint physical-FD force-holdout shards with provenance checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from ase.io import read, write


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labelled_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("shards must use LABEL=/path/to/summary.json")
    label, raw_path = specification.split("=", 1)
    return label, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=labelled_path, action="append", required=True)
    parser.add_argument("--expected-indices", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-xyz", type=Path, required=True)
    args = parser.parse_args()

    expected = sorted(int(value) for value in args.expected_indices.split(","))
    if len(expected) != len(set(expected)):
        raise ValueError("expected indices contain duplicates")
    common_keys = (
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
    merged_atoms = []
    merged_records = []
    observed: list[int] = []
    provenance = []
    reference_settings = None
    for label, summary_path in args.shard:
        payload = json.loads(summary_path.read_text())
        xyz_path = summary_path.with_suffix(".xyz")
        atoms = read(xyz_path, index=":")
        ordered_payload_indices = [int(value) for value in payload["indices"]]
        payload_indices = sorted(ordered_payload_indices)
        if len(atoms) != len(ordered_payload_indices):
            raise ValueError(
                f"shard {label} contains {len(atoms)} XYZ frames for "
                f"{len(ordered_payload_indices)} JSON indices"
            )
        reconstructed_indices = False
        for atom, expected_index in zip(atoms, ordered_payload_indices, strict=True):
            stored_index = atom.info.get("snapshot_index")
            if stored_index is None:
                match = re.fullmatch(
                    r"fd_force_conv_snapshot_(\d+)", str(atom.info.get("config_type", ""))
                )
                if match is None or int(match.group(1)) != expected_index:
                    raise ValueError(
                        f"shard {label} cannot reconstruct snapshot index "
                        f"{expected_index} from XYZ config_type"
                    )
                atom.info["snapshot_index"] = expected_index
                reconstructed_indices = True
            elif int(stored_index) != expected_index:
                raise ValueError(
                    f"shard {label} XYZ order disagrees with JSON indices: "
                    f"{stored_index} != {expected_index}"
                )
        xyz_indices = sorted(int(atom.info["snapshot_index"]) for atom in atoms)
        if payload_indices != xyz_indices:
            raise ValueError(f"shard {label} JSON/XYZ indices disagree")
        settings = {key: payload[key] for key in common_keys}
        if reference_settings is None:
            reference_settings = settings
        elif settings != reference_settings:
            raise ValueError(f"shard {label} calculation settings differ")
        for atom in atoms:
            if str(atom.info.get("split")) != "validation":
                raise ValueError(f"shard {label} contains a non-validation structure")
        observed.extend(payload_indices)
        merged_atoms.extend(atoms)
        merged_records.extend(payload["records"])
        provenance.append(
            {
                "label": label,
                "summary_json": str(summary_path),
                "summary_json_sha256": sha256(summary_path),
                "summary_xyz": str(xyz_path),
                "summary_xyz_sha256": sha256(xyz_path),
                "indices": payload_indices,
                "snapshot_indices_reconstructed_from_config_type": reconstructed_indices,
                "total_wall_seconds": payload["total_wall_seconds"],
            }
        )
    if len(observed) != len(set(observed)):
        raise ValueError(f"holdout shards overlap: {observed}")
    if sorted(observed) != expected:
        raise ValueError(f"holdout union {sorted(observed)} does not match {expected}")

    merged_atoms.sort(key=lambda atom: int(atom.info["snapshot_index"]))
    merged_records.sort(key=lambda item: (int(item["snapshot_index"]), int(item["kgrid"])))
    args.output_xyz.parent.mkdir(parents=True, exist_ok=True)
    temporary_xyz = args.output_xyz.with_name(args.output_xyz.name + ".tmp")
    write(temporary_xyz, merged_atoms, format="extxyz")
    os.replace(temporary_xyz, args.output_xyz)
    result = {
        "status": "complete",
        "role": "fixed independent 600 K force holdout; no training or checkpoint selection",
        "indices": expected,
        "n_structures": len(merged_atoms),
        **(reference_settings or {}),
        "records": merged_records,
        "shards": provenance,
        "output_xyz": str(args.output_xyz),
        "output_xyz_sha256": sha256(args.output_xyz),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = args.output_json.with_name(args.output_json.name + ".tmp")
    temporary_json.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary_json, args.output_json)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
