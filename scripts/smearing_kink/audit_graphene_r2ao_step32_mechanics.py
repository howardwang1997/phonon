#!/usr/bin/env python3
"""Run the proven R2X mechanics audit on the 129-column R2AO checkpoint."""

from __future__ import annotations

import json

import audit_graphene_r2x_paired_mechanics as base
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256
from graphene_r2ao_step32_runtime import load_step32_spectral_checkpoint


CANDIDATE_ROOT = (
    base.BASE / "R2AO_step32_all132_spectral_candidate_retry2_20260827"
)
OUTPUT_ROOT = base.BASE / "R2AO_step32_mechanics_retry2_20260827"
CHECKPOINT_SHA256 = "5e8cbd7c4b59dd486de6b029b05795d321ad3b815e3168f64dae9e41ddb1e8b5"


def main() -> None:
    base.DEFAULT_FREEZE_ROOT = CANDIDATE_ROOT
    base.DEFAULT_OUTPUT = OUTPUT_ROOT
    base.EXPECTED_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    base.load_paired_readout_checkpoint = load_step32_spectral_checkpoint
    base.main()

    summary_path = OUTPUT_ROOT / "summary.json"
    summary = json.loads(summary_path.read_text())
    passed = summary.get("status") == "R2X_PAIRED_READOUT_MECHANICS_PASSED"
    summary.update(
        {
            "format": "graphene_r2ao_step32_spectral_candidate_mechanics_v1",
            "status": (
                "R2AO_SPECTRAL_CANDIDATE_MECHANICS_PASSED"
                if passed
                else "R2AO_SPECTRAL_CANDIDATE_MECHANICS_FAILED"
            ),
            "force_gate_approved": False,
            "authorized_use": "SSCHA spectral sensitivity only",
            "runtime_engine": "coefficient-width-general R2X conservative runtime",
            "candidate_summary": {
                "path": str(CANDIDATE_ROOT / "summary.json"),
                "sha256": file_sha256(CANDIDATE_ROOT / "summary.json"),
            },
        }
    )
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    (OUTPUT_ROOT / "DONE").write_text(summary["status"] + "\n")
    print(json.dumps({"R2AO_status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
