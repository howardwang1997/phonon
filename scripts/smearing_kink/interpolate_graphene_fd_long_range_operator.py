#!/usr/bin/env python3
"""Interpolate the two frozen provisional real-space operators at one temperature."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from conditioned_mace import temperature_weights  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_operator(path: Path, expected_temperature: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "temperature_K",
            "degauss_Ry",
            "delta_fc_full",
            "reference_positions",
            "cell",
            "atom_mapping",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        payload = {key: np.asarray(data[key]) for key in required}
    temperature = float(payload["temperature_K"].reshape(()))
    expected_degauss = 0.0019000869 * expected_temperature / 300.0
    degauss = float(payload["degauss_Ry"].reshape(()))
    if not np.isclose(temperature, expected_temperature, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{path} temperature is {temperature:g}, expected {expected_temperature}")
    if not np.isclose(degauss, expected_degauss, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{path} is not on the fixed physical-FD path")
    force_constants = np.asarray(payload["delta_fc_full"], float)
    if force_constants.ndim != 4 or force_constants.shape[-2:] != (3, 3):
        raise ValueError(f"invalid delta_fc_full shape in {path}: {force_constants.shape}")
    if not np.isfinite(force_constants).all():
        raise ValueError(f"non-finite delta_fc_full in {path}")
    payload["delta_fc_full"] = force_constants
    return payload


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-300", type=Path, required=True)
    parser.add_argument("--operator-600", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    lower = load_operator(args.operator_300, 300)
    upper = load_operator(args.operator_600, 600)
    for key, tolerance in (("cell", 1.0e-12), ("reference_positions", 1.0e-12)):
        if not np.allclose(lower[key], upper[key], rtol=0.0, atol=tolerance):
            raise ValueError(f"endpoint operator {key} arrays differ")
    if not np.array_equal(lower["atom_mapping"], upper["atom_mapping"]):
        raise ValueError("endpoint operator atom mappings differ")
    if lower["delta_fc_full"].shape != upper["delta_fc_full"].shape:
        raise ValueError("endpoint operator force-constant shapes differ")

    weights = temperature_weights(args.temperature)
    force_constants = (
        weights.delta_300 * lower["delta_fc_full"]
        + weights.delta_600 * upper["delta_fc_full"]
    )
    degauss = 0.0019000869 * args.temperature / 300.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    source_300_sha = sha256(args.operator_300)
    source_600_sha = sha256(args.operator_600)
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            temperature_K=np.array(args.temperature),
            degauss_Ry=np.array(degauss),
            interpolation_weight_300=np.array(weights.delta_300),
            interpolation_weight_600=np.array(weights.delta_600),
            delta_fc_full=force_constants,
            reference_positions=np.asarray(lower["reference_positions"], float),
            cell=np.asarray(lower["cell"], float),
            atom_mapping=np.asarray(lower["atom_mapping"], int),
            source_operator_300=np.array(str(args.operator_300)),
            source_operator_600=np.array(str(args.operator_600)),
            source_operator_300_sha256=np.array(source_300_sha),
            source_operator_600_sha256=np.array(source_600_sha),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)

    pair_asymmetry = float(
        np.max(np.abs(force_constants - force_constants.transpose(1, 0, 3, 2)))
    )
    payload = {
        "status": "complete",
        "role": (
            "provisional 450 K sampling operator; subtracted exactly before "
            "short-range TDEP and not used as the final q-space correction"
        ),
        "temperature_K": args.temperature,
        "degauss_Ry": degauss,
        "weights": {
            "operator_300": weights.delta_300,
            "operator_600": weights.delta_600,
        },
        "sources": {
            "300": {
                "path": str(args.operator_300),
                "sha256": source_300_sha,
            },
            "600": {
                "path": str(args.operator_600),
                "sha256": source_600_sha,
            },
        },
        "force_constants_shape": list(force_constants.shape),
        "force_constants_pair_symmetry_max_abs_eV_A2": pair_asymmetry,
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
    }
    atomic_json(args.manifest, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
