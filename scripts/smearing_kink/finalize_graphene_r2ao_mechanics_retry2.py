#!/usr/bin/env python3
"""Finalize R2AO mechanics after repairing a diagnostic-only array order.

The retry2 checkpoint changes only ``train_predicted_force_eV_A``.  Runtime
physics depends on the coefficient, selected columns, and feature scaler,
which must be byte-identical to retry1.  This script binds the already
completed full-Hessian/O(3)/FD audit to that physical-state identity and
reruns the six runtime force replays against the corrected thermal92 order.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256
from graphene_r2ao_step32_runtime import (
    load_step32_spectral_checkpoint,
    production_paired_readout_energy_force,
)
from graphene_r2r0_formal import load_endpoint
from train_graphene_r2s_conditional_mlp import recommended_inputs


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
RETRY1 = BASE / "R2AO_step32_all132_spectral_candidate_retry1_20260827/frozen_readout.npz"
RETRY2_ROOT = BASE / "R2AO_step32_all132_spectral_candidate_retry2_20260827"
RETRY2 = RETRY2_ROOT / "frozen_readout.npz"
PREVIOUS_ROOT = BASE / "R2AO_step32_mechanics_20260827"
OUTPUT = BASE / "R2AO_step32_mechanics_retry2_20260827"
RETRY2_SHA256 = "5e8cbd7c4b59dd486de6b029b05795d321ad3b815e3168f64dae9e41ddb1e8b5"
PHYSICS_MEMBERS = (
    "physical_coefficient",
    "base65_physical_coefficient",
    "bilinear64_physical_coefficient",
    "selected_bilinear_indices",
    "feature_mean",
    "feature_scale",
)


def main() -> None:
    started = time.perf_counter()
    previous_summary_path = PREVIOUS_ROOT / "summary.json"
    previous_arrays_path = PREVIOUS_ROOT / "mechanics_arrays.npz"
    previous = json.loads(previous_summary_path.read_text())
    expected_checks = dict(previous["checks"])
    if previous.get("status") != "R2X_PAIRED_READOUT_MECHANICS_FAILED":
        raise ValueError("R2AO previous audit status changed")
    if expected_checks.pop("runtime_replay", None) is not False or not all(
        expected_checks.values()
    ):
        raise ValueError("R2AO previous audit had a non-replay physics failure")
    if file_sha256(previous_arrays_path) != previous["arrays_sha256"]:
        raise ValueError("R2AO previous mechanics arrays changed")

    with np.load(RETRY1, allow_pickle=False) as left, np.load(
        RETRY2, allow_pickle=False
    ) as right:
        identity = {
            name: bool(np.array_equal(left[name], right[name]))
            for name in PHYSICS_MEMBERS
        }
        retry2_expected = np.asarray(
            right["train_predicted_force_eV_A"], dtype=np.float64
        )
    if not all(identity.values()) or retry2_expected.shape != (92, 72, 3):
        raise ValueError("R2AO retry2 changed runtime physics or replay schema")

    checkpoint = load_step32_spectral_checkpoint(
        RETRY2, expected_sha256=RETRY2_SHA256
    )
    inputs = recommended_inputs()
    model = load_endpoint(inputs, "cpu")
    reference = read(inputs.reference_6x6, index=0)
    thermal = read(inputs.thermal92, index=":")
    replay_indices = (0, 19, 20, 55, 56, 91)
    replay_records = []
    replay_max = 0.0
    for index in replay_indices:
        with torch.enable_grad():
            probe = production_paired_readout_energy_force(
                model,
                thermal[index],
                reference,
                checkpoint,
                device="cpu",
            )
        force = probe.force_source_order_eV_A.detach().cpu().numpy()
        difference = float(np.max(np.abs(force - retry2_expected[index])))
        replay_max = max(replay_max, difference)
        replay_records.append(
            {
                "global_index": index,
                "energy_eV": float(probe.energy_eV.detach().cpu()),
                "force_replay_max_abs_eV_A": difference,
            }
        )
    if replay_max > 1.0e-10:
        raise ValueError(f"R2AO retry2 runtime replay failed: {replay_max:.3e}")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    output_arrays = OUTPUT / "mechanics_arrays.npz"
    shutil.copyfile(previous_arrays_path, output_arrays)
    checks = {**expected_checks, "runtime_replay": True}
    summary = {
        **previous,
        "format": "graphene_r2ao_step32_spectral_candidate_mechanics_v2_identity_reuse",
        "status": "R2AO_SPECTRAL_CANDIDATE_MECHANICS_PASSED",
        "force_gate_approved": False,
        "authorized_use": "SSCHA spectral sensitivity only",
        "checks": checks,
        "runtime_replay": {
            "indices": list(replay_indices),
            "records": replay_records,
            "force_max_abs_difference_eV_A": replay_max,
        },
        "physical_state_identity_retry1_to_retry2": identity,
        "diagnostic_only_change": (
            "train_predicted_force_eV_A reordered/subset from all132 assembly to "
            "the native thermal92 structure order"
        ),
        "reused_full_mechanics_receipt": {
            "summary": {
                "path": str(previous_summary_path),
                "sha256": file_sha256(previous_summary_path),
            },
            "arrays": {
                "path": str(previous_arrays_path),
                "sha256": file_sha256(previous_arrays_path),
            },
            "justification": (
                "all runtime-physics checkpoint members are array-identical; only "
                "the external replay target member changed"
            ),
        },
        "input_sha256": {
            **previous["input_sha256"],
            "checkpoint": RETRY2_SHA256,
            "candidate_summary": file_sha256(RETRY2_ROOT / "summary.json"),
        },
        "arrays_sha256": file_sha256(output_arrays),
        "elapsed_seconds_retry2_finalize": time.perf_counter() - started,
    }
    (OUTPUT / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (OUTPUT / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps({"status": summary["status"], "runtime_replay_max": replay_max}, indent=2))


if __name__ == "__main__":
    main()
