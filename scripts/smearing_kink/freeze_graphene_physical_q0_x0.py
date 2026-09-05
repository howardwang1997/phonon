#!/usr/bin/env python3
"""Freeze the post-L0 Q0/X0 development protocol without opening validation data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


TEMPERATURES = (300, 450, 600)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_passed_json(path: Path, *, bootstrap: bool = False) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "passed":
        raise ValueError(f"required L0 result did not pass: {path}")
    gate_key = "passes_sampling_bootstrap_gate" if bootstrap else "passes_sampling_gate"
    if payload.get(gate_key) is not True:
        raise ValueError(f"required L0 gate is false in {path}")
    return payload


def record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path)}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l0-root", type=Path, required=True)
    parser.add_argument("--q0-root", type=Path, required=True)
    parser.add_argument("--x0-root", type=Path, required=True)
    parser.add_argument("--q0-script", type=Path, required=True)
    parser.add_argument("--x0-analysis-script", type=Path, required=True)
    parser.add_argument("--sampling-script", type=Path, required=True)
    parser.add_argument("--recompute-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (args.l0_root / "L0_ALL_TEMPERATURES_PASSED").exists():
        raise ValueError("L0 has not passed all fixed development gates")
    l0_freeze_path = args.l0_root / "freeze_manifest.json"
    l0_freeze = json.loads(l0_freeze_path.read_text())
    if l0_freeze.get("status") != "frozen_for_L0_development":
        raise ValueError("unexpected L0 freeze-manifest status")
    if l0_freeze.get("development_temperatures_K") != list(TEMPERATURES):
        raise ValueError("unexpected L0 development temperatures")
    if l0_freeze.get("locked_validation_temperatures_K_not_accessed") != [375, 525]:
        raise ValueError("the locked validation boundary is missing")

    l0_results = {}
    for temperature in TEMPERATURES:
        lane = args.l0_root / f"T{temperature}"
        point = lane / "final_acceptance.json"
        bootstrap = lane / "final_bootstrap_acceptance.json"
        read_passed_json(point)
        read_passed_json(bootstrap, bootstrap=True)
        l0_results[str(temperature)] = {
            "point_gate": record(point),
            "bootstrap_gate": record(bootstrap),
            "physical_tdep": record(lane / "final_physical_tdep.npz"),
            "pooled_short_tdep": record(lane / "final_short_pooled.npz"),
            "seed0_short_tdep": record(lane / "final_short_seed0.npz"),
        }

    scripts = {
        "q0_sscha": record(args.q0_script),
        "x0_analysis": record(args.x0_analysis_script),
        "x0_fallback_sampling": record(args.sampling_script),
        "x0_fallback_recompute": record(args.recompute_script),
    }
    payload = {
        "status": "frozen_for_Q0_X0_development",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post-L0 Q0 quantum and X0 electronic-lattice cross-term development",
        "development_temperatures_K": list(TEMPERATURES),
        "locked_validation_temperatures_K_not_accessed": [375, 525],
        "validation_release": "forbidden; stop at READY_FOR_VALIDATION_FREEZE_REVIEW",
        "source_L0": {
            "freeze_manifest": record(l0_freeze_path),
            "results": l0_results,
            "base_model": l0_freeze["base_model"],
            "delta_model": l0_freeze["delta_model"],
            "background": l0_freeze["background"],
            "operators": l0_freeze["operators"],
        },
        "paths": {
            "q0_root": str(args.q0_root),
            "x0_root": str(args.x0_root),
        },
        "Q0_protocol": {
            "execution_order_K": [450, 300, 600],
            "benchmark": {
                "lattice_temperature_K": 450,
                "operator_temperature_K": 450,
                "n_configs": 200,
                "max_populations": 4,
                "random_seed": 450004,
            },
            "formal": {
                "temperatures_K": [450, 300, 600],
                "n_configs": 300,
                "max_populations": 5,
                "random_seeds": {"300": 300005, "450": 450005, "600": 600005},
            },
            "fixed_gate": {
                "SSCHA_relax_returns_converged": True,
                "all_frequencies_finite": True,
                "maximum_population_is_not_automatically_accepted": True,
            },
        },
        "X0_protocol": {
            "first_non_diagonal_condition": {
                "lattice_temperature_K": 450,
                "operator_temperature_K": 300,
                "smearing_formula": "degauss (Ry) = k_B*300 K/Ry",
                "n_configs": 300,
                "max_populations": 5,
                "random_seed": 453005,
            },
            "formula_reconstruction": (
                "Phi_Q0(Tlat,Tlat) - Phi_operator(Tlat) + Phi_operator(Tel)"
            ),
            "fixed_gate": {
                "top_branch_max_abs_difference_cm-1": 2.0,
                "K_kink_relative_change": 0.20,
                "SSCHA_relax_returns_converged": True,
            },
            "on_gate_pass": "reconstruct the remaining 3x3 development grid by formula",
            "on_gate_failure": (
                "run one fixed-seed classical-MD diagnostic at (Tlat=450 K, Tel=300 K), "
                "then stop before any three-seed expansion"
            ),
            "fallback_MD": {
                "seed": 0,
                "dt_fs": 0.5,
                "equilibration_steps": 3000,
                "stride": 80,
                "snapshots": 3000,
                "checkpoint_every_saved_snapshots": 25,
            },
        },
        "scripts": scripts,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
