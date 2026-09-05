#!/usr/bin/env python3
"""Check completeness and q/-q symmetry of a full static EPW q grid."""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def load_qpoints(path: Path, ngrid: int) -> list[dict]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    header = lines[0].split()
    if len(header) != 2 or int(header[0]) != ngrid * ngrid or header[1] != "crystal":
        raise ValueError(f"unexpected q-point header: {lines[0]}")
    result = []
    for iq, line in enumerate(lines[1:], 1):
        fields = [float(value) for value in line.split()]
        if len(fields) != 4:
            raise ValueError(f"unexpected q-point row: {line}")
        key = tuple(int(round(value * ngrid)) % ngrid for value in fields[:2])
        result.append({"iq": iq, "q": fields[:3], "weight": fields[3], "key": key})
    if len(result) != ngrid * ngrid:
        raise ValueError(f"expected {ngrid * ngrid} q points, found {len(result)}")
    if len({row["key"] for row in result}) != len(result):
        raise ValueError("q grid contains duplicate points")
    if not math.isclose(sum(row["weight"] for row in result), 1.0, abs_tol=1.0e-9):
        raise ValueError("q-point weights do not sum to one")
    return result


def load_static_rows(path: Path) -> dict[tuple[int, int], dict]:
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 9:
            raise ValueError(f"unexpected specfun_sup.phon row: {line}")
        iq, mode = int(fields[0]), int(fields[1])
        values = [float(value) for value in fields[2:]]
        if abs(values[3]) > 1.0e-12:
            continue
        key = (iq, mode)
        if key in result:
            raise ValueError(f"duplicate zero-frequency row {key}")
        result[key] = {
            "temperature_K": values[0],
            "broadening_eV": values[1],
            "bare_eV": values[2],
            "delta_pi_meV": values[4] - values[5],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpoints", type=Path, required=True)
    parser.add_argument("--static-self-energy", type=Path, required=True)
    parser.add_argument("--ngrid", type=int, default=9)
    parser.add_argument("--temperature-K", type=float, default=450.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    qpoints = load_qpoints(args.qpoints, args.ngrid)
    rows = load_static_rows(args.static_self_energy)
    expected_rows = len(qpoints) * 6
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} zero-frequency rows, found {len(rows)}")
    q_by_key = {tuple(row["key"]): row for row in qpoints}
    max_bare_pair_eV = 0.0
    max_delta_pi_pair_meV = 0.0
    for qpoint in qpoints:
        inverse = ((-qpoint["key"][0]) % args.ngrid, (-qpoint["key"][1]) % args.ngrid)
        paired = q_by_key[inverse]
        left = sorted(
            (rows[(qpoint["iq"], mode)] for mode in range(1, 7)),
            key=lambda row: row["bare_eV"],
        )
        right = sorted(
            (rows[(paired["iq"], mode)] for mode in range(1, 7)),
            key=lambda row: row["bare_eV"],
        )
        for row_left, row_right in zip(left, right, strict=True):
            max_bare_pair_eV = max(
                max_bare_pair_eV, abs(row_left["bare_eV"] - row_right["bare_eV"])
            )
            max_delta_pi_pair_meV = max(
                max_delta_pi_pair_meV,
                abs(row_left["delta_pi_meV"] - row_right["delta_pi_meV"]),
            )

    temperatures = {round(row["temperature_K"], 8) for row in rows.values()}
    broadenings = {round(row["broadening_eV"], 8) for row in rows.values()}
    checks = {
        "row_count": len(rows) == expected_rows,
        "temperature_matches_target": temperatures
        == {round(float(args.temperature_K), 8)},
        "broadening_is_0p005eV": broadenings == {0.005},
        "q_minus_q_bare_max_le_1e-6_eV": max_bare_pair_eV <= 1.0e-6,
        "q_minus_q_delta_pi_max_le_1e-3_meV": max_delta_pi_pair_meV <= 1.0e-3,
    }
    ready = all(checks.values())
    payload = {
        "status": "ready" if ready else "invalid",
        "scope": (
            f"complete q{args.ngrid} static EPW data integrity; "
            "not the Cartesian operator gate"
        ),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "ngrid": args.ngrid,
        "target_temperature_K": float(args.temperature_K),
        "n_qpoints": len(qpoints),
        "n_zero_frequency_rows": len(rows),
        "max_q_minus_q_bare_difference_eV": max_bare_pair_eV,
        "max_q_minus_q_delta_pi_difference_meV": max_delta_pi_pair_meV,
        "checks": checks,
        "releases_E1": False,
        "next_stage": (
            "construct_cartesian_Hermitian_operator_and_run_real_space_replay"
            if ready
            else "stop_and_diagnose_full_q_data"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(
        args.output_dir / "full_q_data_audit.json",
        json.dumps(payload, indent=2) + "\n",
    )
    marker = args.output_dir / (
        "FULL_Q_DATA_READY" if ready else "FULL_Q_DATA_INVALID"
    )
    atomic_text(marker, payload["audited_at_utc"] + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
