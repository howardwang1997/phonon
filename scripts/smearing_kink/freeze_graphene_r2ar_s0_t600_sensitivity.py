#!/usr/bin/env python3
"""Freeze an independent S0 fixed-smearing SSCHA sensitivity case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
SOURCE = BASE / "S0_unified_short/Q0_X0_freeze_manifest.json"
RUNNER = ROOT / "scripts/smearing_kink/run_graphene_physical_q0_sscha.py"
SHARED_INITIAL_RUNNER = (
    ROOT / "scripts/smearing_kink/run_graphene_r2at_s0_shared_initial_sscha.py"
)
DEFAULT_OUTPUT = (
    BASE
    / "R2R_multipolar_background/R2AR_s0_T600_fixed_smearing_sensitivity_manifest_20260827.json"
)
EXPECTED_SOURCE_SHA256 = (
    "77de1d23cbf33e3557de1a6bdc728274155fda32fc96f5597e759b912d66a383"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--lattice-temperature", type=int, choices=(300, 450, 600), default=600
    )
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--n-configs", type=int, default=300)
    parser.add_argument("--shared-initial", action="store_true")
    args = parser.parse_args()
    temperature = int(args.lattice_temperature)
    random_seed = (
        int(args.random_seed)
        if args.random_seed is not None
        else {300: 300005, 450: 450005, 600: 600005}[temperature]
    )
    if args.n_configs <= 0 or random_seed < 0:
        raise ValueError("n-configs must be positive and random-seed nonnegative")
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise ValueError("archived Q0/X0 freeze manifest changed")
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_for_Q0_X0_development":
        raise ValueError("archived Q0/X0 status changed")
    if manifest.get("locked_validation_temperatures_K_not_accessed") != [375, 525]:
        raise ValueError("locked validation boundary changed")

    manifest["created_at_utc"] = "2026-08-27T00:00:00+00:00"
    pairing = " with the R2AP shared initial Hessian" if args.shared_initial else ""
    manifest["scope"] = (
        f"independent S0 short-range model sensitivity at Tlat={temperature} K and "
        f"fixed degauss corresponding to the frozen 300 K operator{pairing}"
    )
    variant_stage = "R2AT" if args.shared_initial else "R2AR"
    manifest["manifest_variant"] = (
        f"{variant_stage}_S0_T{temperature}_FIXED_SMEARING_SPECTRAL_SENSITIVITY"
    )
    manifest["sensitivity_extension"] = {
        "authorized_use": "compare the R2AO and independent S0 SSCHA spectra",
        "not_a_replacement_for_R2AO_force_validation": True,
        "lattice_temperature_K": temperature,
        "operator_temperature_K": 300,
        "fixed_smearing_degauss_Ry": 0.0019000869380739254,
        "n_configs": int(args.n_configs),
        "max_populations": 5,
        "random_seed": random_seed,
        "paired_initial_hessian": bool(args.shared_initial),
        "unseen_525K_access": False,
    }
    manifest["X0_protocol"]["first_non_diagonal_condition"] = {
        "lattice_temperature_K": temperature,
        "operator_temperature_K": 300,
        "smearing_formula": "degauss (Ry) = k_B*300 K/Ry",
        "n_configs": int(args.n_configs),
        "max_populations": 5,
        "random_seed": random_seed,
    }
    manifest["scripts"]["q0_sscha"]["sha256"] = sha256(RUNNER)
    root_name = (
        "R2AT_s0_shared_initial_sensitivity"
        if args.shared_initial
        else "R2AR_s0_fixed_smearing_sensitivity"
    )
    manifest["paths"]["x0_root"] = (
        "/home/howardwang/phonon/results/graphene_physics_temperature/"
        "post_p4_feasibility/R2R_multipolar_background/" + root_name
    )
    if args.shared_initial:
        wrapper_record = {
            "path": str(SHARED_INITIAL_RUNNER.relative_to(ROOT)),
            "sha256": sha256(SHARED_INITIAL_RUNNER),
        }
        manifest["scripts"]["r2at_shared_initial_sscha"] = wrapper_record
        manifest["sensitivity_extension"].update(
            {
                "shared_initial_runner": wrapper_record,
                "shared_initial_formula": (
                    "Phi_TDEP(Tlat,Tel=Tlat) - Phi_q6(Tel=Tlat) "
                    "+ Phi_q6(Tel=300K)"
                ),
                "paired_sampling_gate": (
                    "R2AO and S0 xats_pop1 arrays must agree within 1e-12 A"
                ),
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_bytes(canonical_bytes(manifest) + b"\n")
    print(
        json.dumps(
            {
                "status": manifest["manifest_variant"],
                "path": str(output),
                "sha256": sha256(output),
                "runner_sha256": manifest["scripts"]["q0_sscha"]["sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
