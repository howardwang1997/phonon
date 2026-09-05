#!/usr/bin/env python3
"""Verify and summarize exact-setting TMD cold-vs-Fermi-Dirac fc2 pairs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INDIR = ROOT / "results" / "tmd_exp_a_recovery"
CM_PER_THZ = 33.35641
SYSTEMS = (
    ("NbSe2", "3.440"),
    ("NbS2", "3.320"),
    ("2H-TaSe2", "3.436"),
    ("1T-VSe2", "3.340"),
    ("1T-TiSe2", "3.540"),
)
MATCHED_FIELDS = (
    "material",
    "a",
    "thickness",
    "supercell",
    "ecutwfc",
    "ecutrho",
    "kpts",
    "displacement",
    "degauss",
)


def load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def scalar(value: np.ndarray):
    result = value.item()
    return result.item() if hasattr(result, "item") else result


def path_location(data: dict[str, np.ndarray], index: int) -> str:
    distance = float(np.asarray(data["distances"])[index])
    ticks = np.asarray(data["label_positions"], float)
    labels = [str(value) for value in np.asarray(data["labels"])]
    segment = int(np.clip(np.searchsorted(ticks, distance, side="right") - 1, 0, len(ticks) - 2))
    denominator = ticks[segment + 1] - ticks[segment]
    fraction = 0.0 if abs(denominator) < 1.0e-12 else (distance - ticks[segment]) / denominator
    return f"{labels[segment]}-{labels[segment + 1]}:{fraction:.4f}"


def main() -> int:
    rows = []
    for material, lattice_slug in SYSTEMS:
        prefix = f"{material}_exp_a{lattice_slug}_dg0.005"
        fd_path = INDIR / f"disp_{prefix}_fd.npz"
        cold_path = INDIR / f"disp_{prefix}_cold.npz"
        fd, cold = load(fd_path), load(cold_path)

        mismatches = {}
        for field in MATCHED_FIELDS:
            left, right = scalar(fd[field]), scalar(cold[field])
            if isinstance(left, (float, np.floating)):
                matched = np.isclose(float(left), float(right), rtol=0.0, atol=1.0e-12)
            else:
                matched = left == right
            if not matched:
                mismatches[field] = {"fd": left, "cold": right}
        for field in ("distances", "label_positions", "labels"):
            left, right = np.asarray(fd[field]), np.asarray(cold[field])
            matched = np.array_equal(left, right) if left.dtype.kind in "OUS" else np.allclose(left, right)
            if not matched:
                mismatches[field] = "array mismatch"
        if str(scalar(fd["smearing"])) != "fd" or str(scalar(cold["smearing"])) != "cold":
            mismatches["smearing"] = {
                "fd": str(scalar(fd["smearing"])),
                "cold": str(scalar(cold["smearing"])),
            }
        if mismatches:
            raise RuntimeError(f"{material} is not an exact cold/fd pair: {mismatches}")

        fd_freq = np.asarray(fd["frequencies"], float)
        cold_freq = np.asarray(cold["frequencies"], float)
        if fd_freq.shape != cold_freq.shape or not (
            np.isfinite(fd_freq).all() and np.isfinite(cold_freq).all()
        ):
            raise RuntimeError(f"invalid paired frequency arrays for {material}")
        fd_flat = int(np.argmin(fd_freq))
        cold_flat = int(np.argmin(cold_freq))
        fd_q, fd_branch = np.unravel_index(fd_flat, fd_freq.shape)
        cold_q, cold_branch = np.unravel_index(cold_flat, cold_freq.shape)
        fd_min = float(fd_freq[fd_q, fd_branch] * CM_PER_THZ)
        cold_min = float(cold_freq[cold_q, cold_branch] * CM_PER_THZ)
        rows.append(
            {
                "material": material,
                "a_Angstrom": float(scalar(fd["a"])),
                "supercell": int(scalar(fd["supercell"])),
                "degauss_Ry": float(scalar(fd["degauss"])),
                "fd_min_cm-1": fd_min,
                "cold_min_cm-1": cold_min,
                "cold_minus_fd_min_cm-1": cold_min - fd_min,
                "fd_soft_q": path_location(fd, fd_q),
                "cold_soft_q": path_location(cold, cold_q),
                "fd_soft_branch": int(fd_branch + 1),
                "cold_soft_branch": int(cold_branch + 1),
                "fd_n_imag_below_-0.1THz": int(np.count_nonzero(fd_freq < -0.1)),
                "cold_n_imag_below_-0.1THz": int(np.count_nonzero(cold_freq < -0.1)),
            }
        )

    summary = {
        "status": "complete",
        "pair_definition": (
            "identical material, lattice, thickness, supercell, cutoff, k grid, "
            "displacement and degauss; only QE smearing='fd' versus 'cold' differs"
        ),
        "n_pairs": len(rows),
        "rows": rows,
    }
    with (INDIR / "tmd_cold_fd_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (INDIR / "tmd_cold_fd_pairs.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
