"""Independent 450 K DFT-TDEP reference and acceptance vs the deployed curve.

Fits an effective harmonic (TDEP-style hiPhive) force-constant model on the
fixed-smearing DFT labels of the 450 K matched closure, evaluates the A'
branch on the deployed 241-point near-K q mesh, and compares against the
deployed S0 full-EPC 450 K curve with the frozen closure gates:

  * K A' absolute difference                 <= 5 cm-1
  * K-referenced shape RMSE / max            <= 1 / 2 cm-1
  * cusp depth relative error (d=0.003, 0.025) <= 15 %
  * split-half K A' spread                   <= 5 cm-1

Inputs are the per-batch label directories written by
``graphene_fd_force_convergence.py`` plus the frozen snapshot npz files
(atom order: phonopy supercell of the background yaml, verified upstream).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
import td_phonon as tdp  # noqa: E402
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402

BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
R2R = BASE / "R2R_multipolar_background"
DEPLOYED = (
    R2R
    / "R2AT_paired_full_epc_acceptance_20260827/r2ao_s0_fixed_smearing_full_epc_matrices.npz"
)
BACKGROUND = ROOT / "results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
OUT = BASE / "R2A_450k_dft_tdep_reference"

CUTOFF2 = 6.0
APRIME_BRANCH = 5
CUSP_OFFSETS = (0.003, 0.025)
GATES = {
    "K_Aprime_abs_diff_cm1": 5.0,
    "K_referenced_shape_RMSE_cm1": 1.0,
    "K_referenced_shape_max_cm1": 2.0,
    "cusp_depth_rel_error": 0.15,
    "split_half_K_Aprime_spread_cm1": 5.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(batch_dirs: list[Path]) -> list[dict]:
    records = []
    for directory in batch_dirs:
        snapshots = directory / "snapshots.npz"
        with np.load(snapshots, allow_pickle=False) as data:
            sscha_indices = np.asarray(data["sscha_indices"], int)
            positions_all = np.asarray(data["positions"], float)
        for local_index, sscha_index in enumerate(sscha_indices):
            label_path = directory / "labels" / f"snapshot_{local_index:03d}_k8.npz"
            with np.load(label_path, allow_pickle=False) as data:
                forces = np.asarray(data["forces"], float)
                energy = float(data["energy"])
                positions = np.asarray(data["positions"], float)
            if not np.allclose(positions, positions_all[local_index], atol=1e-8):
                raise ValueError(f"positions differ for sscha {sscha_index}")
            records.append(
                {
                    "sscha_index": int(sscha_index),
                    "positions": positions,
                    "forces": forces,
                    "energy": energy,
                    "source": str(directory),
                }
            )
    indices = [r["sscha_index"] for r in records]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate sscha indices across batches")
    return records


def fit_fc2(phonon, ideal_positions, cell, records) -> tuple[np.ndarray, dict]:
    inv_cell = np.linalg.inv(cell)
    structures = []
    for record in records:
        delta = record["positions"] - ideal_positions
        fractional = delta @ inv_cell
        fractional -= np.round(fractional)
        aligned = ideal_positions + fractional @ cell
        atoms = Atoms(
            numbers=[6] * len(ideal_positions),
            positions=aligned,
            cell=cell,
            pbc=True,
        )
        atoms.calc = SinglePointCalculator(atoms, forces=record["forces"])
        structures.append(atoms)
    primitive = phonopy_to_ase(phonon.primitive)
    primitive.wrap()
    sc_matrix = np.asarray(phonon.supercell_matrix, int)
    fc2, rmse, n_dof = tdp.effective_fc2(primitive, phonopy_to_ase(phonon.supercell), sc_matrix, structures, CUTOFF2)
    return fc2, {"fit_rmse_eV_A": rmse, "n_dof": n_dof, "n_structures": len(structures)}


def aprime_curve(phonon, fc2, qpoints) -> np.ndarray:
    phonon.force_constants = fc2
    phonon.run_qpoints(qpoints)
    frequencies = phonon.get_qpoints_dict()["frequencies"]
    frequencies = np.asarray(frequencies, float)
    top = frequencies[:, APRIME_BRANCH]
    second = frequencies[:, APRIME_BRANCH - 1]
    if not np.all(top > second):
        raise ValueError("A' branch is not the top branch along the whole window")
    return top * 33.35641


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="append", type=Path, required=True,
                        help="batch dir containing snapshots.npz and labels/")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    records = collect(args.batch)
    phonon = fm.load_ph(BACKGROUND)
    ideal_positions = np.asarray(phonon.supercell.positions, float)
    cell = np.asarray(phonon.supercell.cell, float)

    fc2, fit_info = fit_fc2(phonon, ideal_positions, cell, records)

    with np.load(DEPLOYED, allow_pickle=False) as data:
        deployed_s0 = np.asarray(data["full_EPC_frequency_cm1"])[1, 1, :]
        deployed_r2ao = np.asarray(data["full_EPC_frequency_cm1"])[0, 1, :]
        qpoints = np.asarray(data["qpoints"], float)
        signed_q = np.asarray(data["signed_q_2pi_over_a"], float)
        assert np.asarray(data["lattice_temperature_K"])[1] == 450

    k_index = int(np.argmin(np.abs(signed_q)))
    reference = aprime_curve(phonon, fc2, qpoints)

    def cusp_depths(curve: np.ndarray) -> dict[str, float]:
        depths = {}
        for offset in CUSP_OFFSETS:
            left = np.interp(-offset, signed_q, curve)
            right = np.interp(offset, signed_q, curve)
            depths[f"d{offset}"] = float(0.5 * (left + right) - curve[k_index])
        return depths

    ref_depths = cusp_depths(reference)
    dep_depths = cusp_depths(deployed_s0)

    shape_ref = reference - reference[k_index]
    shape_dep = deployed_s0 - deployed_s0[k_index]
    shape_delta = shape_ref - shape_dep

    order = np.argsort([r["sscha_index"] for r in records])
    ordered = [records[i] for i in order]
    half_a = ordered[0::2]
    half_b = ordered[1::2]
    fc2_a, _ = fit_fc2(phonon, ideal_positions, cell, half_a)
    fc2_b, _ = fit_fc2(phonon, ideal_positions, cell, half_b)
    k_a = aprime_curve(phonon, fc2_a, qpoints[k_index : k_index + 1])[0]
    k_b = aprime_curve(phonon, fc2_b, qpoints[k_index : k_index + 1])[0]

    metrics = {
        "n_labels": len(records),
        **fit_info,
        "K_Aprime_DFT_TDEP_cm1": float(reference[k_index]),
        "K_Aprime_deployed_S0_cm1": float(deployed_s0[k_index]),
        "K_Aprime_deployed_R2AO_cm1": float(deployed_r2ao[k_index]),
        "K_Aprime_abs_diff_cm1": float(abs(reference[k_index] - deployed_s0[k_index])),
        "K_referenced_shape_RMSE_cm1": float(np.sqrt((shape_delta**2).mean())),
        "K_referenced_shape_max_cm1": float(np.abs(shape_delta).max()),
        "cusp_depth_DFT_TDEP_cm1": ref_depths,
        "cusp_depth_deployed_cm1": dep_depths,
        "cusp_depth_rel_error": {
            key: abs(ref_depths[key] - dep_depths[key]) / abs(dep_depths[key])
            for key in ref_depths
        },
        "split_half_K_Aprime_cm1": [float(k_a), float(k_b)],
        "split_half_K_Aprime_spread_cm1": float(abs(k_a - k_b)),
    }
    gates = {
        "K_Aprime_abs_diff_le_5": metrics["K_Aprime_abs_diff_cm1"] <= GATES["K_Aprime_abs_diff_cm1"],
        "shape_RMSE_le_1": metrics["K_referenced_shape_RMSE_cm1"] <= GATES["K_referenced_shape_RMSE_cm1"],
        "shape_max_le_2": metrics["K_referenced_shape_max_cm1"] <= GATES["K_referenced_shape_max_cm1"],
        "cusp_depth_le_15pct": all(
            value <= GATES["cusp_depth_rel_error"]
            for value in metrics["cusp_depth_rel_error"].values()
        ),
        "split_half_spread_le_5": metrics["split_half_K_Aprime_spread_cm1"] <= GATES["split_half_K_Aprime_spread_cm1"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "dft_tdep_reference.npz",
        fc2_eV_A2=fc2,
        qpoints=qpoints,
        signed_q_2pi_over_a=signed_q,
        aprime_frequency_cm1=reference,
        deployed_s0_aprime_frequency_cm1=deployed_s0,
        deployed_r2ao_aprime_frequency_cm1=deployed_r2ao,
        sscha_indices=np.array(sorted(r["sscha_index"] for r in records), int),
        cutoff2=np.array(CUTOFF2),
        fit_rmse_eV_A=np.array(fit_info["fit_rmse_eV_A"]),
        n_dof=np.array(fit_info["n_dof"]),
        fc2_split_a=fc2_a,
        fc2_split_b=fc2_b,
    )
    payload = {
        "status": "all_gates_passed" if all(gates.values()) else "gate_failure",
        "scope": "independent 450 K fixed-smearing DFT-TDEP reference vs deployed S0 full-EPC curve",
        "condition": {
            "lattice_temperature_K": 450.0,
            "degauss_Ry": 0.0019000869380739254,
            "background": str(BACKGROUND),
            "background_sha256": sha256(BACKGROUND),
            "deployed_matrices": str(DEPLOYED),
            "deployed_matrices_sha256": sha256(DEPLOYED),
            "cutoff2_A": CUTOFF2,
            "aprime_branch": APRIME_BRANCH,
        },
        "gates_frozen": GATES,
        "metrics": metrics,
        "gates": gates,
        "batches": [str(batch) for batch in args.batch],
    }
    (args.output_dir / "acceptance.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"metrics": metrics, "gates": gates}, indent=2))
    print(f"wrote {args.output_dir / 'acceptance.json'}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
