#!/usr/bin/env python3
"""Compare graphene two-band Wannier Hamiltonians near the Dirac point."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_hr(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    num_wann = int(lines[1])
    nrpts = int(lines[2])
    ndeg_lines = (nrpts + 14) // 15
    degeneracies = np.asarray(
        [int(value) for line in lines[3 : 3 + ndeg_lines] for value in line.split()],
        float,
    )
    if len(degeneracies) != nrpts:
        raise ValueError(f"{path}: expected {nrpts} degeneracies, found {len(degeneracies)}")

    records = lines[3 + ndeg_lines :]
    if len(records) != nrpts * num_wann * num_wann:
        raise ValueError(f"{path}: unexpected Hamiltonian record count")
    translations = np.empty((nrpts, 3), int)
    matrices = np.zeros((nrpts, num_wann, num_wann), complex)
    cursor = 0
    for ir in range(nrpts):
        for _ in range(num_wann * num_wann):
            fields = records[cursor].split()
            cursor += 1
            r = tuple(int(value) for value in fields[:3])
            m = int(fields[3]) - 1
            n = int(fields[4]) - 1
            translations[ir] = r
            matrices[ir, m, n] = complex(float(fields[5]), float(fields[6]))
        matrices[ir] /= degeneracies[ir]
    return translations, matrices


def eigenvalues(
    translations: np.ndarray, matrices: np.ndarray, kpoints: np.ndarray
) -> np.ndarray:
    result = []
    for kpoint in kpoints:
        phase = np.exp(2j * np.pi * (translations @ kpoint))
        hamiltonian = np.einsum("r,rij->ij", phase, matrices)
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
        result.append(np.linalg.eigvalsh(hamiltonian))
    return np.asarray(result, float)


def case_summary(path: Path, fermi_eV: float, t_values: np.ndarray) -> tuple[dict, np.ndarray]:
    translations, matrices = read_hr(path)
    kpoints = np.column_stack((t_values / 3.0, t_values / 3.0, np.zeros_like(t_values)))
    bands = eigenvalues(translations, matrices, kpoints)
    k_bands = eigenvalues(
        translations, matrices, np.asarray([[1.0 / 3.0, 1.0 / 3.0, 0.0]])
    )[0]
    dirac_mid = float(np.mean(k_bands))

    mask = (np.abs(t_values - 1.0) >= 0.002) & (np.abs(t_values - 1.0) <= 0.02)
    distance_t = np.abs(t_values[mask] - 1.0)
    conduction = bands[mask, 1] - dirac_mid
    valence = dirac_mid - bands[mask, 0]
    slope_conduction = float(np.dot(distance_t, conduction) / np.dot(distance_t, distance_t))
    slope_valence = float(np.dot(distance_t, valence) / np.dot(distance_t, distance_t))
    gamma_k_distance_per_t_invA = 4.0 * np.pi / (3.0 * 2.46)

    return (
        {
            "source": str(path),
            "fermi_eV": fermi_eV,
            "gamma_bands_eV": [float(value) for value in bands[0]],
            "K_bands_eV": [float(value) for value in k_bands],
            "K_gap_eV": float(k_bands[1] - k_bands[0]),
            "dirac_mid_eV": dirac_mid,
            "dirac_mid_minus_fermi_eV": dirac_mid - fermi_eV,
            "vF_conduction_eVA": slope_conduction / gamma_k_distance_per_t_invA,
            "vF_valence_eVA": slope_valence / gamma_k_distance_per_t_invA,
            "wannier_R_count": int(len(translations)),
        },
        bands,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--old-fermi", type=float, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--matched-fermi", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    t_values = np.linspace(0.0, 1.04, 521)
    old_summary, old_bands = case_summary(args.old, args.old_fermi, t_values)
    matched_summary, matched_bands = case_summary(args.matched, args.matched_fermi, t_values)
    old_relative = old_bands - old_summary["dirac_mid_eV"]
    matched_relative = matched_bands - matched_summary["dirac_mid_eV"]

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "old_epw3": old_summary,
        "matched_15A_k12q6": matched_summary,
        "relative_band_difference": {
            "RMSE_eV": float(np.sqrt(np.mean((matched_relative - old_relative) ** 2))),
            "max_abs_eV": float(np.max(np.abs(matched_relative - old_relative))),
        },
    }
    (output / "wannier_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    with (output / "wannier_GK_bands.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["t_GK", "old_band1_rel_eV", "old_band2_rel_eV", "matched_band1_rel_eV", "matched_band2_rel_eV"]
        )
        for index, t_value in enumerate(t_values):
            writer.writerow([t_value, *old_relative[index], *matched_relative[index]])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
