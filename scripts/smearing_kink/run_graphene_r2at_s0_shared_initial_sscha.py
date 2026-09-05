#!/usr/bin/env python3
"""Run an S0 fixed-smearing SSCHA case paired to the R2AO initial Hessian.

This wrapper deliberately reuses the frozen Q0/X0 runner and changes only its
initial-force-constant callback.  The trial Hessian is exactly the one used by
R2AP at the same lattice temperature::

    Phi_TDEP(Tlat, Tel=Tlat) - Phi_q6(Tel=Tlat) + Phi_q6(Tel=300 K)

With the same NumPy seed this gives R2AO and S0 a common initial distribution,
so their first-population configurations can be checked pair by pair.  The
wrapper and the unchanged Q0/X0 runner are both hash-bound by the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for item in (HERE, ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_graphene_physical_q0_sscha as q0  # noqa: E402


def resolve_record(record: dict) -> Path:
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=True)
    if q0.sha256(path) != record["sha256"]:
        raise ValueError(f"R2AT frozen input hash changed: {path}")
    return path


def shared_initial_force_constants(
    freeze: dict,
    phase: str,
    lattice_temperature: int,
    operator_temperature: int,
) -> tuple[np.ndarray, dict]:
    if phase != "x0_first" or operator_temperature != 300:
        raise ValueError("R2AT shared-initial runner only supports fixed-T300 x0_first")
    record = freeze["source_L0"]["results"][str(lattice_temperature)][
        "physical_tdep"
    ]
    path = resolve_record(record)
    with np.load(path, allow_pickle=False) as data:
        physical = np.asarray(data["pooled_fc2"], dtype=np.float64)
    _, matched_operator, _, _, _ = q0.operator_fc(freeze, lattice_temperature)
    _, fixed_operator, _, _, _ = q0.operator_fc(freeze, operator_temperature)
    if physical.shape != matched_operator.shape or physical.shape != fixed_operator.shape:
        raise ValueError("R2AT shared initial/operator force-constant shapes differ")
    return physical - matched_operator + fixed_operator, {
        "kind": "R2AT_shared_matched_TDEP_minus_matched_q6_plus_fixed_q6",
        "physical_tdep": {**record, "resolved_path": str(path)},
        "matched_operator_temperature_K": lattice_temperature,
        "fixed_operator_temperature_K": operator_temperature,
        "formula": (
            "Phi_TDEP(Tlat,Tel=Tlat) - Phi_q6(Tel=Tlat) "
            "+ Phi_q6(Tel=300K)"
        ),
        "paired_with_R2AP_initial_hessian": True,
    }


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--freeze-manifest", type=Path, required=True)
    known, _ = pre_parser.parse_known_args()
    manifest_path = known.freeze_manifest.resolve(strict=True)
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    extension = freeze.get("sensitivity_extension", {})
    if extension.get("paired_initial_hessian") is not True:
        raise ValueError("R2AT manifest does not authorize the shared initial Hessian")
    wrapper_record = extension.get("shared_initial_runner")
    if not isinstance(wrapper_record, dict):
        raise ValueError("R2AT manifest lacks the shared-initial runner record")
    if resolve_record(wrapper_record) != Path(__file__).resolve():
        raise ValueError("R2AT manifest points to a different shared-initial runner")
    q0.initial_force_constants = shared_initial_force_constants
    return q0.main()


if __name__ == "__main__":
    raise SystemExit(main())
