"""Audit the synced graphene P0 ``gr.dyn`` development data.

The audit verifies file hashes, path metadata, QE eigenvectors, Hermiticity and
matrix-to-frequency reconstruction against the archived strict CSVs.
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
INDIR = ROOT / "results" / "p0_graphene_dfpt" / "dyn"
OUTDIR = ROOT / "results" / "p0_graphene_qspace"


def csv_reference() -> dict[tuple[str, float, int, str, float], np.ndarray]:
    rows = {}
    for lane in ("A", "B"):
        path = ROOT / "results" / "p0_graphene_dfpt" / f"dfpt_linecuts_{lane}.csv"
        with path.open() as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["lane"],
                    float(row["degauss_Ry"]),
                    int(row["kgrid"]),
                    row["region"],
                    float(row["t_GK"]),
                )
                rows[key] = np.array([float(row[f"f{i}_cm"]) for i in range(1, 7)])
    return rows


def metadata(path: Path) -> tuple[str, float, int, str, float]:
    rel = path.relative_to(INDIR)
    lane = rel.parts[0]
    set_match = re.fullmatch(r"dg(.+)_k(\d+)", rel.parts[1])
    q_match = re.fullmatch(r"([GK])_t(.+)", rel.parts[2])
    if not set_match or not q_match:
        raise ValueError(f"unexpected development path: {path}")
    return (
        lane,
        float(set_match.group(1)),
        int(set_match.group(2)),
        q_match.group(1),
        float(q_match.group(2).replace("p", ".")),
    )


def main() -> int:
    refs = csv_reference()
    files = sorted(INDIR.glob("*/*/*/gr.dyn"))
    if len(files) != 56:
        raise RuntimeError(f"expected 56 gr.dyn files, found {len(files)}")

    records = []
    for path in files:
        key = metadata(path)
        dyn = load_qe_dyn(path)
        if key not in refs:
            raise RuntimeError(f"no CSV reference for {key}")
        csv_error = float(np.max(np.abs(dyn.frequencies_cm - refs[key])))
        matrix_freq = dyn.frequencies_from_matrix_cm()
        matrix_error = float(np.max(np.abs(matrix_freq - dyn.frequencies_cm)))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "lane": key[0],
                "degauss_Ry": key[1],
                "kgrid": key[2],
                "region": key[3],
                "t_GK": key[4],
                "q": dyn.q_cart_2pi_over_a.tolist(),
                "sha256": sha,
                "bytes": path.stat().st_size,
                "hermitian_max_abs": dyn.hermitian_error,
                "eigenvector_orthogonality_max_abs": dyn.eigenvector_orthogonality_error,
                "csv_frequency_max_abs_cm-1": csv_error,
                "matrix_frequency_max_abs_cm-1": matrix_error,
                "matrix_scale_cm-2": dyn.matrix_frequency_scale_cm2(),
            }
        )

    summary = {
        "status": "complete",
        "n_files": len(records),
        "n_q_k32": sum(r["kgrid"] == 32 for r in records),
        "n_q_k64": sum(r["kgrid"] == 64 for r in records),
        "max_hermitian_error": max(r["hermitian_max_abs"] for r in records),
        "max_eigenvector_orthogonality_error": max(
            r["eigenvector_orthogonality_max_abs"] for r in records
        ),
        "max_csv_frequency_error_cm-1": max(
            r["csv_frequency_max_abs_cm-1"] for r in records
        ),
        "max_matrix_frequency_error_cm-1": max(
            r["matrix_frequency_max_abs_cm-1"] for r in records
        ),
        "matrix_scale_cm-2_min": min(r["matrix_scale_cm-2"] for r in records),
        "matrix_scale_cm-2_max": max(r["matrix_scale_cm-2"] for r in records),
    }
    if summary["max_hermitian_error"] > 5e-7:
        raise RuntimeError(f"Hermiticity audit failed: {summary}")
    if summary["max_eigenvector_orthogonality_error"] > 5e-5:
        raise RuntimeError(f"eigenvector audit failed: {summary}")
    if summary["max_csv_frequency_error_cm-1"] > 1e-6:
        raise RuntimeError(f"CSV replay audit failed: {summary}")
    # gr.dyn prints matrix elements to only eight decimal places; the replay
    # tolerance is therefore looser than the parsed-frequency audit.
    if summary["max_matrix_frequency_error_cm-1"] > 0.1:
        raise RuntimeError(f"matrix replay audit failed: {summary}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "development_dyn_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUTDIR / "development_dyn_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
