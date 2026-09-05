#!/usr/bin/env python3
"""Strictly audit the supplemental DEV_A/DEV_B direct-DFPT campaigns.

The final HOLD_A/HOLD_B directories are deliberately not accepted by this
script.  Sync each completed campaign to
``results/p0_graphene_qspace/development_dfpt/<campaign>`` before running it.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from qe_dyn import load_qe_dyn


ROOT = Path(__file__).resolve().parents[2]
INDIR = ROOT / "results" / "p0_graphene_qspace" / "development_dfpt"
CAMPAIGNS = {"DEV_A": (0.02, 14), "DEV_B": (0.06, 14)}


def q_key(region: str, value: float) -> tuple[str, float]:
    return region, round(float(value), 12)


def main() -> int:
    records = []
    for campaign, (expected_dg, expected_count) in CAMPAIGNS.items():
        campaign_dir = INDIR / campaign
        csv_path = campaign_dir / f"dfpt_{campaign}.csv"
        done_path = campaign_dir / "DONE"
        if not done_path.is_file() or not csv_path.is_file():
            raise RuntimeError(f"incomplete local campaign mirror: {campaign_dir}")

        references = {}
        with csv_path.open() as handle:
            for row in csv.DictReader(handle):
                if (
                    row["campaign"] != campaign
                    or not np.isclose(float(row["degauss_Ry"]), expected_dg)
                    or int(row["kgrid"]) != 32
                ):
                    raise RuntimeError(f"unexpected row in {csv_path}: {row}")
                key = q_key(row["region"], float(row["t_GK"]))
                if key in references:
                    raise RuntimeError(f"duplicate row {key} in {csv_path}")
                references[key] = np.array(
                    [float(row[f"f{i}_cm"]) for i in range(1, 7)]
                )
        if len(references) != expected_count:
            raise RuntimeError(
                f"{campaign}: expected {expected_count} CSV rows, "
                f"found {len(references)}"
            )

        dyn_paths = sorted(campaign_dir.glob("dg*_k32/[GK]_t*/gr.dyn"))
        if len(dyn_paths) != expected_count:
            raise RuntimeError(
                f"{campaign}: expected {expected_count} gr.dyn files, "
                f"found {len(dyn_paths)}"
            )
        for path in dyn_paths:
            set_match = re.fullmatch(r"dg(.+)_k32", path.parents[1].name)
            q_match = re.fullmatch(r"([GK])_t(.+)", path.parent.name)
            if not set_match or not q_match:
                raise RuntimeError(f"unexpected path: {path}")
            dg = float(set_match.group(1))
            t_value = float(q_match.group(2).replace("p", "."))
            key = q_key(q_match.group(1), t_value)
            if not np.isclose(dg, expected_dg) or key not in references:
                raise RuntimeError(f"unregistered development matrix: {path}")

            dyn = load_qe_dyn(path)
            csv_error = float(
                np.max(np.abs(dyn.frequencies_cm - references[key]))
            )
            matrix_error = float(
                np.max(
                    np.abs(dyn.frequencies_from_matrix_cm() - dyn.frequencies_cm)
                )
            )
            records.append(
                {
                    "campaign": campaign,
                    "degauss_Ry": dg,
                    "kgrid": 32,
                    "region": key[0],
                    "t_GK": t_value,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                    "hermitian_max_abs": dyn.hermitian_error,
                    "eigenvector_orthogonality_max_abs": (
                        dyn.eigenvector_orthogonality_error
                    ),
                    "csv_frequency_max_abs_cm-1": csv_error,
                    "matrix_frequency_max_abs_cm-1": matrix_error,
                }
            )

    summary = {
        "status": "complete",
        "scope": "supplemental development only; no HOLD campaign read",
        "campaigns": list(CAMPAIGNS),
        "n_files": len(records),
        "max_hermitian_error": max(row["hermitian_max_abs"] for row in records),
        "max_eigenvector_orthogonality_error": max(
            row["eigenvector_orthogonality_max_abs"] for row in records
        ),
        "max_csv_frequency_error_cm-1": max(
            row["csv_frequency_max_abs_cm-1"] for row in records
        ),
        "max_matrix_frequency_error_cm-1": max(
            row["matrix_frequency_max_abs_cm-1"] for row in records
        ),
    }
    if summary["max_hermitian_error"] > 5e-7:
        raise RuntimeError(f"Hermiticity audit failed: {summary}")
    if summary["max_eigenvector_orthogonality_error"] > 5e-5:
        raise RuntimeError(f"eigenvector audit failed: {summary}")
    if summary["max_csv_frequency_error_cm-1"] > 1e-6:
        raise RuntimeError(f"CSV replay audit failed: {summary}")
    if summary["max_matrix_frequency_error_cm-1"] > 0.1:
        raise RuntimeError(f"matrix replay audit failed: {summary}")

    with (INDIR / "development_campaign_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (INDIR / "development_campaign_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
