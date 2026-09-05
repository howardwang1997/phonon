#!/usr/bin/env python3
"""Validate the frozen finite-lattice inputs for the no-degauss K-cusp run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "configs/graphene_k_cusp_nosmear/data_inventory.tsv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def expected_shape(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split("x"))


def scalar(payload: np.lib.npyio.NpzFile, key: str) -> object:
    return np.asarray(payload[key]).item()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    with args.inventory.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 12:
        raise ValueError(f"expected 12 frozen inputs, found {len(rows)}")

    records: list[dict] = []
    payloads: dict[tuple[int, str], dict] = {}
    for row in rows:
        temperature = int(row["temperature_K"])
        channel = row["channel"]
        path = args.root / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        if actual_size != int(row["bytes"]):
            raise ValueError(f"size mismatch for {path}: {actual_size}")
        actual_hash = digest(path)
        if actual_hash != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")
        with np.load(path, allow_pickle=False) as archive:
            key = row["npz_key"]
            if key not in archive.files:
                raise KeyError(f"missing {key} in {path}")
            shape = tuple(np.asarray(archive[key]).shape)
            if shape != expected_shape(row["shape"]):
                raise ValueError(f"shape mismatch for {path}:{key}: {shape}")
            metadata = {
                name: scalar(archive, name)
                for name in (
                    "temperature_K",
                    "lattice_temperature_K",
                    "operator_temperature_K",
                    "degauss_Ry",
                    "operator_sha256",
                    "converged",
                )
                if name in archive.files
            }
        payloads[(temperature, channel)] = metadata
        records.append(
            {
                "temperature_K": temperature,
                "channel": channel,
                "role": row["role"],
                "path": row["path"],
                "npz_key": row["npz_key"],
                "shape": list(shape),
                "bytes": actual_size,
                "sha256": actual_hash,
                "metadata": metadata,
            }
        )

    for temperature in (300, 450, 600):
        channels = {channel for temp, channel in payloads if temp == temperature}
        if channels != {"static", "L0", "Q0", "q6_operator"}:
            raise ValueError(f"incomplete T={temperature} input group: {channels}")
        l0 = payloads[(temperature, "L0")]
        q0 = payloads[(temperature, "Q0")]
        operator = payloads[(temperature, "q6_operator")]
        operator_record = next(
            record
            for record in records
            if record["temperature_K"] == temperature
            and record["channel"] == "q6_operator"
        )
        if int(l0["temperature_K"]) != temperature:
            raise ValueError(f"L0 temperature mismatch at T={temperature}")
        if l0["operator_sha256"] != operator_record["sha256"]:
            raise ValueError(f"L0/operator provenance mismatch at T={temperature}")
        if not bool(q0["converged"]):
            raise ValueError(f"Q0 is not converged at T={temperature}")
        if int(q0["lattice_temperature_K"]) != temperature:
            raise ValueError(f"Q0 lattice temperature mismatch at T={temperature}")
        if int(q0["operator_temperature_K"]) != temperature:
            raise ValueError(f"Q0 operator temperature mismatch at T={temperature}")
        if int(operator["lattice_temperature_K"]) != temperature:
            raise ValueError(f"q6 operator temperature mismatch at T={temperature}")

    result = {
        "status": "passed",
        "scope": "frozen finite-lattice inputs for graphene no-degauss K cusp",
        "inventory": str(args.inventory),
        "inventory_sha256": digest(args.inventory),
        "n_files": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "temperatures_K": [300, 450, 600],
        "records": records,
    }
    print(json.dumps(result, indent=2, default=lambda value: value.item()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
