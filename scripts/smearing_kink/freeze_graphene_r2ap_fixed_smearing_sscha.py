#!/usr/bin/env python3
"""Freeze inputs and protocol for the R2AO fixed-smearing SSCHA queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_OUTPUT = BASE / "R2AP_fixed_smearing_SSCHA_manifest_20260827.json"
RUNNER = ROOT / "scripts/smearing_kink/run_graphene_r2ap_fixed_smearing_sscha.py"
CANDIDATE = BASE / "R2AO_step32_all132_spectral_candidate_retry2_20260827"
MECHANICS = BASE / "R2AO_step32_mechanics_retry2_20260827/summary.json"


def record(relative: str) -> dict[str, str]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "sha256": file_sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    mechanics = json.loads(MECHANICS.read_text())
    if mechanics.get("status") != "R2AO_SPECTRAL_CANDIDATE_MECHANICS_PASSED":
        raise ValueError("R2AP cannot freeze before R2AO mechanics passes")
    candidate_summary = json.loads((CANDIDATE / "summary.json").read_text())
    if (
        candidate_summary.get("status")
        != "R2AO_FROZEN_FOR_SPECTRAL_SENSITIVITY_MECHANICS_PENDING"
        or candidate_summary.get("force_gate_approved") is not False
    ):
        raise ValueError("R2AP candidate boundary changed")

    operator_root = (
        "results/graphene_physics_temperature/post_p4_feasibility/"
        "E0_epw_matched/k18_q9_ex1_pifroz/operator_q6_450K/cartesian"
    )
    tdep_root = (
        "results/graphene_physics_temperature/post_p4_feasibility/"
        "S0_unified_short/L0_classical_tdep"
    )
    operators = {}
    smearings = {
        300: 0.0019000869380739254,
        450: 0.0028501304071108886,
        600: 0.003800173876147851,
    }
    for temperature in (300, 450, 600):
        operators[str(temperature)] = {
            **record(f"{operator_root}/T{temperature}_operator.npz"),
            "degauss_Ry": smearings[temperature],
        }
    initial = {
        str(temperature): record(
            f"{tdep_root}/T{temperature}/final_physical_tdep.npz"
        )
        for temperature in (300, 450, 600)
    }
    source_paths = {
        "R2AO_runtime": "scripts/smearing_kink/graphene_r2ao_step32_runtime.py",
        "R2X_conservative_engine": "scripts/smearing_kink/graphene_r2x_paired_readout.py",
        "R2R_background_engine": "scripts/smearing_kink/graphene_r2r_multipolar_background.py",
        "R2AG_schema": "scripts/smearing_kink/train_graphene_r2ag_seed012_conditional_mlp.py",
        "R2O_Taylor_engine": "scripts/smearing_kink/graphene_r2o_taylor_null.py",
        "Q0_conversion_helpers": "scripts/smearing_kink/run_graphene_physical_q0_sscha.py",
    }
    manifest = {
        "format": "graphene_r2ap_fixed_smearing_SSCHA_manifest_v1",
        "status": "R2AP_FIXED_SMEARING_SSCHA_FROZEN",
        "runner": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": file_sha256(RUNNER),
        "source_files": {
            name: record(path) for name, path in source_paths.items()
        },
        "protocol": {
            "temperatures_K": [300, 450, 600],
            "execution_order_K": [450, 300, 600],
            "n_configs": 300,
            "max_populations": 5,
            "random_seeds": {"300": 300005, "450": 450005, "600": 600005},
            "acceptance": {
                "SSCHA_relax_returns_converged": True,
                "all_outputs_finite": True,
                "imaginary_modes_below_minus_1_cm-1": 0,
            },
        },
        "fixed_operator_temperature_K": 300,
        "fixed_smearing_degauss_Ry": smearings[300],
        "lattice_temperature_and_electronic_smearing_are_distinct": True,
        "foundation_model": record("results/gr_backbone_v11/ft_graphene.model"),
        "background": record(
            "results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
        ),
        "operators": operators,
        "initial_physical_tdep": initial,
        "R2AO_checkpoint": record(
            str(
                (
                    CANDIDATE / "spectral_candidate_readout.npz"
                ).relative_to(ROOT)
            )
        ),
        "R2AO_candidate_summary": record(
            str((CANDIDATE / "summary.json").relative_to(ROOT))
        ),
        "R2AO_mechanics_summary": record(str(MECHANICS.relative_to(ROOT))),
        "R2Q_endpoint": {
            "checkpoint": record(
                "results/graphene_physics_temperature/post_p4_feasibility/"
                "R2Q_four_step_trust_region/formal_4step_seed83_rtx/endpoint.pt"
            ),
            "receipt": record(
                "results/graphene_physics_temperature/post_p4_feasibility/"
                "R2Q_four_step_trust_region/formal_4step_seed83_rtx/endpoint_receipt.json"
            ),
            "marker": record(
                "results/graphene_physics_temperature/post_p4_feasibility/"
                "R2Q_four_step_trust_region/formal_4step_seed83_rtx/ENDPOINT_FROZEN"
            ),
        },
        "model_validation_boundary": {
            "force_gate_approved": False,
            "authorized_use": "spectral sensitivity only",
            "mandatory_followup": [
                "compare SSCHA spectra against an independent short-range candidate",
                "replace q6 interpolation by the frozen dense full-EPC K-line response",
                "apply quantitative Kohn-cusp depth/width/frequency gates",
            ],
        },
        "energy_labels_used_in_R2AO": False,
        "unseen_525K_access": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_bytes(canonical_json_bytes(manifest) + b"\n")
    print(json.dumps({"status": manifest["status"], "path": str(output), "sha256": file_sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
