#!/usr/bin/env python3
"""Verify a copied E0 operator bundle before materializing an E1 release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann, electron_volt, physical_constants


RYDBERG_EV = physical_constants["Rydberg constant times hc in eV"][0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--operator-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decision = json.loads(args.decision.read_text())
    if decision.get("status") != "passed" or decision.get("releases_E1") is not True:
        raise ValueError("aggregate operator decision does not release E1")
    if decision.get("development_temperatures_K") != [300, 450, 600]:
        raise ValueError("unexpected development temperatures")
    if decision.get("locked_validation_temperatures_K_not_accessed") != [375, 525]:
        raise ValueError("locked validation boundary is missing")
    records = decision.get("operators", [])
    if [int(record["temperature_K"]) for record in records] != [300, 450, 600]:
        raise ValueError("operator records are incomplete or out of order")

    verified = []
    reference_hashes, cell_hashes = set(), set()
    for record in records:
        temperature = int(record["temperature_K"])
        path = args.operator_dir / f"T{temperature}_operator.npz"
        if sha256(path) != record["operator_sha256"]:
            raise ValueError(f"copied T{temperature} operator hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            force_constants = np.asarray(data["delta_fc_full"], float)
            reference = np.asarray(data["reference_positions"], float)
            cell = np.asarray(data["cell"], float)
            degauss = float(data["degauss_Ry"])
            stored_temperature = float(data["temperature_K"])
        expected = Boltzmann * temperature / electron_volt / RYDBERG_EV
        if abs(stored_temperature - temperature) > 1.0e-8:
            raise ValueError(f"stored temperature mismatch at T{temperature}")
        if abs(degauss - expected) > 1.0e-12:
            raise ValueError(f"degauss formula mismatch at T{temperature}")
        if force_constants.shape != (72, 72, 3, 3):
            raise ValueError(f"force-constant shape mismatch at T{temperature}")
        if not np.isfinite(force_constants).all():
            raise ValueError(f"non-finite force constant at T{temperature}")
        pair_error = float(
            np.max(np.abs(force_constants - force_constants.transpose(1, 0, 3, 2)))
        )
        asr_error = float(np.max(np.abs(force_constants.sum(axis=1))))
        if pair_error > 1.0e-8 or asr_error > 1.0e-8:
            raise ValueError(f"force gate failed after copy at T{temperature}")
        reference_hashes.add(hashlib.sha256(reference.tobytes()).hexdigest())
        cell_hashes.add(hashlib.sha256(cell.tobytes()).hexdigest())
        verified.append(
            {
                "temperature_K": temperature,
                "degauss_Ry": degauss,
                "operator_sha256": record["operator_sha256"],
                "force_pair_max_abs_eV_A2": pair_error,
                "force_ASR_max_abs_eV_A2": asr_error,
            }
        )
    if len(reference_hashes) != 1 or len(cell_hashes) != 1:
        raise ValueError("reference geometry is inconsistent across copied operators")

    payload = {
        "status": "verified_for_E1_release",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_sha256": sha256(args.decision),
        "operators": verified,
        "locked_validation_temperatures_K_not_accessed": [375, 525],
    }
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
