#!/usr/bin/env python3
"""Fail-closed local launcher for the authorized R2R-1 readout fit.

R2R-1 label parsing and fitting are intentionally local.  This launcher has no
SSH, SCP, subprocess, staging, or remote command path, and therefore cannot
transmit thermal labels or targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import aggregate_graphene_r2r1_linear_readout as aggregate
import graphene_r2r1_linear_readout as linear


FORMAT = "graphene_r2r1_local_launcher_v1"
REMOTE_STAGE_ALLOWLIST: tuple[str, ...] = ()
THERMAL_OR_LABEL_REMOTE_STAGE_ALLOWED = False
DEFAULT_THERMAL92 = (
    linear.ROOT / "data/graphene_r2o_taylor_null_core/train_thermal.xyz"
)


def execute(
    *,
    output_root: Path,
    attempt3_root: Path,
    thermal92_path: Path,
    freeze_manifest: Path,
    authorization_marker: Path,
    release_manifest: Path,
) -> dict[str, Any]:
    """Preflight locally, then fit or recover through the sole aggregate API."""
    # Reject every path mismatch/overlap before preflight or any path that can
    # authorize label access.  Payload existence and hashes are checked later
    # by the state-specific held-fd transaction.
    aggregate._validate_production_path_contract(
        output_root=output_root,
        attempt3_root=attempt3_root,
        thermal92_path=thermal92_path,
        freeze_manifest=freeze_manifest,
        authorization_marker=authorization_marker,
        release_manifest=release_manifest,
        input_existence_required=False,
    )
    release_state = aggregate.inspect_release_state(output_root)
    if release_state["state"] == "DONE":
        release_anchor = aggregate.anchor_completed_release(
            output_root=output_root,
            attempt3_root=attempt3_root,
            thermal92_path=thermal92_path,
            freeze_manifest=freeze_manifest,
            authorization_marker=authorization_marker,
            release_manifest=release_manifest,
        )
        release = {
            "status": release_anchor["release_status"],
            "mechanics_status": release_anchor["mechanics_status"],
            "recovered_through_held_anchor_transaction": True,
        }
        return {
            "format": FORMAT,
            "status": release["status"],
            "preflight_status": "RECOVERED_EXISTING_DONE_RELEASE",
            "release": release,
            "external_release_manifest": release_anchor,
            "external_release_manifest_requires_next_stage_SHA_binding": True,
            "fit_materialization_repeated": False,
            "remote_execution": False,
            "remote_stage_allowlist": [],
            "thermal_or_label_remote_stage_allowed": False,
            "mechanics_execution_in_this_launcher": False,
            "mechanics_status": release["mechanics_status"],
        }
    if release_state["state"] != "ABSENT":
        return {
            "format": FORMAT,
            "status": f"R2R1_EXISTING_{release_state['state']}",
            "preflight_status": "NOT_RUN_EXISTING_RELEASE_STATE",
            "release_state": release_state,
            "fit_started": False,
            "fit_materialization_repeated": False,
            "remote_execution": False,
            "remote_stage_allowlist": [],
            "thermal_or_label_remote_stage_allowed": False,
            "mechanics_execution_in_this_launcher": False,
        }
    publication_pair = aggregate._inspect_fresh_publication_pair(output_root)
    if publication_pair["both_absent"] is not True:
        return {
            "format": FORMAT,
            "status": "R2R1_EXISTING_CANDIDATE_COLLISION",
            "preflight_status": "NOT_RUN_EXISTING_CANDIDATE_STATE",
            "publication_pair": publication_pair,
            "fit_started": False,
            "fit_materialization_repeated": False,
            "remote_execution": False,
            "remote_stage_allowlist": [],
            "thermal_or_label_remote_stage_allowed": False,
            "mechanics_execution_in_this_launcher": False,
        }
    preflight = linear.label_blind_preflight(attempt3_root, thermal92_path)
    if preflight["status"] != "R2R1_LABEL_BLIND_PREFLIGHT_PASSED":
        return {
            "format": FORMAT,
            "status": "R2R1_NUMERICAL_INCONCLUSIVE_PREFLIGHT",
            "preflight": preflight,
            "fit_started": False,
            "remote_execution": False,
            "remote_stage_allowlist": [],
            "thermal_or_label_remote_stage_allowed": False,
        }
    release = aggregate.materialize_authorized_fit(
        output_root=output_root,
        attempt3_root=attempt3_root,
        thermal92_path=thermal92_path,
        freeze_manifest=freeze_manifest,
        authorization_marker=authorization_marker,
    )
    release_anchor = aggregate.anchor_completed_release(
        output_root=output_root,
        attempt3_root=attempt3_root,
        thermal92_path=thermal92_path,
        freeze_manifest=freeze_manifest,
        authorization_marker=authorization_marker,
        release_manifest=release_manifest,
        materialization_witness=release.anchor_witness,
    )
    return {
        "format": FORMAT,
        "status": release["status"],
        "preflight_status": preflight["status"],
        "release": release,
        "external_release_manifest": release_anchor,
        "external_release_manifest_requires_next_stage_SHA_binding": True,
        "remote_execution": False,
        "remote_stage_allowlist": [],
        "thermal_or_label_remote_stage_allowed": False,
        "mechanics_execution_in_this_launcher": False,
        "mechanics_status": release["mechanics_status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt3-root", type=Path, default=linear.ATTEMPT3_ROOT)
    parser.add_argument("--thermal92", type=Path, default=DEFAULT_THERMAL92)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--authorization-marker", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = execute(
        output_root=args.output_root,
        attempt3_root=args.attempt3_root,
        thermal92_path=args.thermal92,
        freeze_manifest=args.freeze_manifest,
        authorization_marker=args.authorization_marker,
        release_manifest=args.release_manifest,
    )
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return (
        0
        if receipt.get("status") in aggregate.SCIENTIFIC_STATUSES
        and "external_release_manifest" in receipt
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
