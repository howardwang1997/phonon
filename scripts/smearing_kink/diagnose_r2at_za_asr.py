"""Diagnose the full-path ZA dips near Gamma in the R2AT acceptance bands.

The acceptance records ZA minima of -2.99/-10.59/-6.40 cm^-1 (300/450/600 K)
on the interpolated M-Gamma-K-M path and attributes them to Fourier
interpolation residues. This script separates three candidate causes:

1. translational invariance (acoustic sum rule) violation of the SSCHA
   free-energy FC2;
2. a physically negative flexural (bending) rigidity of the free-energy
   Hessian, visible already at commensurate q;
3. Fourier-interpolation ringing between commensurate nodes of the 6x6
   supercell force constants.

Findings (2026-09-05): ASR drift is ~1e-13 eV/A^2 and Gamma ZA is exactly
zero, so (1) is excluded. The commensurate spectrum minimum is ~0 and ZA at
the smallest commensurate ring q=(1/6,0) is +67..74 cm^-1, so (2) is
excluded. The negative well appears only at incommensurate q between Gamma
and the first ring, which identifies (3): ringing of the trigonometric
interpolation polynomial. Values below are in cm^-1 (phonopy THz x
33.35641).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

BASE = Path(
    "results/graphene_physics_temperature/post_p4_feasibility/S0_unified_short"
)
RESULT_DIRS = {
    temperature: BASE / "Q0_quantum_sscha" / f"formal_T{temperature}" / "result.npz"
    for temperature in (300, 450, 600)
}
OPERATOR_PATH = Path("data/graphene_r2c_eval/operators/T300_operator.npz")
CARBON_MASS_AMU = 12.011
THZ_TO_CM1 = 33.35641


def make_phonopy(cell: np.ndarray) -> Phonopy:
    primitive = np.array([cell[0] / 6.0, cell[1] / 6.0, cell[2]])
    unitcell = PhonopyAtoms(
        symbols=["C", "C"],
        cell=primitive,
        scaled_positions=[[0.0, 0.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]],
        masses=[CARBON_MASS_AMU, CARBON_MASS_AMU],
    )
    return Phonopy(unitcell, supercell_matrix=np.diag([6, 6, 1]), primitive_matrix=np.eye(3))


def band_qpoints(n_per_segment: int = 180) -> np.ndarray:
    gamma = np.array([0.0, 0.0])
    k_point = np.array([1.0 / 3.0, 1.0 / 3.0])
    m_point = np.array([0.5, 0.0])
    q_list: list[np.ndarray] = []
    for start, end in ((m_point, gamma), (gamma, k_point), (k_point, m_point)):
        for value in np.linspace(0.0, 1.0, n_per_segment, endpoint=False):
            q_list.append(start + value * (end - start))
    q_list.append(m_point.copy())
    return np.array([[q[0], q[1], 0.0] for q in q_list])


def za_along(phonon: Phonopy, qpoints: np.ndarray) -> np.ndarray:
    za = []
    for q in qpoints:
        phonon.run_qpoints([q])
        za.append(np.sort(phonon.get_qpoints_dict()["frequencies"][0])[0] * THZ_TO_CM1)
    return np.asarray(za, float)


def asr_drift(fc: np.ndarray) -> dict:
    row = fc.sum(axis=1)  # sum_j Phi_ij (72, 3, 3)
    mean_block = row.mean(axis=0)  # rigid-translation Gamma block (3, 3)
    return {
        "max_abs_row_sum_eV_A2": float(np.abs(row).max()),
        "rms_row_sum_eV_A2": float(np.sqrt(np.mean(row**2))),
        "rigid_translation_eigenvalues_eV_A2_per_amu": [
            float(v) for v in np.linalg.eigvalsh(mean_block / CARBON_MASS_AMU)
        ],
    }


def main() -> int:
    operator = np.load(OPERATOR_PATH, allow_pickle=True)
    cell = np.asarray(operator["cell"], float)
    qpoints = band_qpoints()
    fine_gk = np.array([[t / 3.0, t / 3.0, 0.0] for t in np.linspace(0.005, 0.5, 200)])
    commensurate = [(1.0 / 6.0, 0.0), (1.0 / 6.0, 1.0 / 6.0)]
    report: dict = {"units": "cm^-1", "temperatures": {}}
    for temperature, path in RESULT_DIRS.items():
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1
        data = np.load(path, allow_pickle=True)
        fc = np.asarray(data["free_energy_fc2_eV_A2"], float)
        stored = np.sort(np.asarray(data["frequency_cm_1"], float), axis=1)[:, 0]
        supercell_min = float(np.sort(np.asarray(data["supercell_frequency_cm_1"], float))[0])

        phonon = make_phonopy(cell)
        raw_asr = asr_drift(fc)
        phonon.force_constants = fc
        za_band = za_along(phonon, qpoints)
        za_fine = za_along(phonon, fine_gk)
        za_commensurate = za_along(
            phonon, np.array([[q[0], q[1], 0.0] for q in commensurate])
        )
        gamma_za = float(za_along(phonon, np.array([[0.0, 0.0, 0.0]]))[0])

        fc_sym = fc.copy()
        phonon.force_constants = fc_sym
        phonon.symmetrize_force_constants()
        sym_asr = asr_drift(np.asarray(phonon.force_constants, float))
        za_band_sym = za_along(phonon, qpoints)

        dip_index = int(np.argmin(za_fine))
        report["temperatures"][str(temperature)] = {
            "stored_result_za_min_cm1": float(stored.min()),
            "commensurate_spectrum_min_cm1": supercell_min,
            "za_at_commensurate_ring_cm1": {
                "(1/6,0)": float(za_commensurate[0]),
                "(1/6,1/6)": float(za_commensurate[1]),
            },
            "gamma_za_cm1": gamma_za,
            "interpolated_za_min_cm1": float(za_fine[dip_index]),
            "interpolated_za_min_towards_K_fraction": float(
                np.linspace(0.005, 0.5, 200)[dip_index]
            ),
            "raw_asr": raw_asr,
            "symmetrized_asr": sym_asr,
            "path_za_min_raw_cm1": float(za_band.min()),
            "path_za_min_symmetrized_cm1": float(za_band_sym.min()),
            "fc_max_abs_change_on_symmetrization_eV_A2": float(
                np.abs(np.asarray(phonon.force_constants, float) - fc).max()
            ),
        }
        entry = report["temperatures"][str(temperature)]
        print(
            f"T{temperature}: stored min {entry['stored_result_za_min_cm1']:8.3f} | "
            f"commensurate min {supercell_min:7.3f} | ring ZA "
            f"{entry['za_at_commensurate_ring_cm1']['(1/6,0)']:7.3f} | "
            f"Gamma ZA {gamma_za:6.3f} | interp dip "
            f"{entry['interpolated_za_min_cm1']:7.3f} at t={entry['interpolated_za_min_towards_K_fraction']:.3f} | "
            f"ASR {raw_asr['max_abs_row_sum_eV_A2']:.2e} -> sym {sym_asr['max_abs_row_sum_eV_A2']:.2e}"
        )
    out = Path(
        "results/graphene_physics_temperature/post_p4_feasibility/"
        "R2R_multipolar_background/R2AT_paired_full_epc_acceptance_20260827/"
        "za_asr_diagnostic.json"
    )
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
