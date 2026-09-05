#!/usr/bin/env python3
"""Aggregate the three development-temperature Cartesian operator gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann, electron_volt, physical_constants


TEMPERATURES = (300, 450, 600)
RYDBERG_EV = physical_constants["Rydberg constant times hc in eV"][0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-dir", type=Path, required=True)
    args = parser.parse_args()

    records = []
    reference_hashes, cell_hashes = set(), set()
    for temperature in TEMPERATURES:
        gate_path = args.operator_dir / f"T{temperature}_operator_gate.json"
        marker = args.operator_dir / f"T{temperature}_OPERATOR_PASS"
        operator_path = args.operator_dir / f"T{temperature}_operator.npz"
        if not marker.is_file():
            raise FileNotFoundError(marker)
        gate = json.loads(gate_path.read_text())
        if gate["status"] != "passed" or gate["releases_E1"] is not False:
            raise ValueError(f"invalid non-releasing T{temperature} gate")
        if int(round(float(gate["temperature_K"]))) != temperature:
            raise ValueError(f"temperature mismatch in {gate_path}")
        if gate["operator"]["sha256"] != sha256(operator_path):
            raise ValueError(f"operator hash mismatch at T{temperature}")
        with np.load(operator_path, allow_pickle=False) as data:
            force_constants = np.asarray(data["delta_fc_full"], float)
            reference = np.asarray(data["reference_positions"], float)
            cell = np.asarray(data["cell"], float)
            degauss = float(data["degauss_Ry"])
            qpoints = np.asarray(data["qpoints_crystal"], float)
        expected_degauss = Boltzmann * temperature / electron_volt / RYDBERG_EV
        if abs(degauss - expected_degauss) > 1.0e-12:
            raise ValueError(f"degauss formula mismatch at T{temperature}")
        if force_constants.shape != (72, 72, 3, 3) or not np.isfinite(force_constants).all():
            raise ValueError(f"invalid force constants at T{temperature}")
        if qpoints.shape != (36, 3):
            raise ValueError(f"invalid q grid at T{temperature}")
        pair_error = float(
            np.max(np.abs(force_constants - force_constants.transpose(1, 0, 3, 2)))
        )
        asr_error = float(np.max(np.abs(force_constants.sum(axis=1))))
        if pair_error > 1.0e-8 or asr_error > 1.0e-8:
            raise ValueError(f"force operator audit failed at T{temperature}")
        reference_hashes.add(hashlib.sha256(reference.tobytes()).hexdigest())
        cell_hashes.add(hashlib.sha256(cell.tobytes()).hexdigest())
        records.append(
            {
                "temperature_K": temperature,
                "degauss_Ry": degauss,
                "operator": str(operator_path),
                "operator_sha256": sha256(operator_path),
                "gate": str(gate_path),
                "gate_sha256": sha256(gate_path),
                "force_constant_frobenius_eV_A2": float(np.linalg.norm(force_constants)),
                "force_pair_max_abs_eV_A2": pair_error,
                "force_ASR_max_abs_eV_A2": asr_error,
            }
        )
    if len(reference_hashes) != 1 or len(cell_hashes) != 1:
        raise ValueError("operator reference geometry changed across temperatures")

    payload = {
        "status": "passed",
        "scope": "E0 300/450/600 K Cartesian Hermitian q6 operator aggregate gate",
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
        "temperature_law": "smearing/degauss (Ry) = k_B*T/Ry",
        "development_temperatures_K": list(TEMPERATURES),
        "locked_validation_temperatures_K_not_accessed": [375, 525],
        "operators": records,
        "reference_geometry_sha256": next(iter(reference_hashes)),
        "cell_sha256": next(iter(cell_hashes)),
        "releases_E1": True,
        "next_stage": "run_30_off_diagonal_cross_degauss_DFT_labels",
    }
    decision = args.operator_dir / "operator_gate_decision.json"
    atomic_text(decision, json.dumps(payload, indent=2) + "\n")
    atomic_text(args.operator_dir / "OPERATOR_GATE_PASS", payload["decided_at_utc"] + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
