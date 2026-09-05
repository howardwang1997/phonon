"""Independent 450 K DFT reference and acceptance vs the deployed curve.

Two estimators, both composed with the shared DFT Kohn machinery:

1. Absolute K A' (primary): tagged-pair thermal curvature.  The deployed
   thermal Hessian is the SSCHA free-energy Hessian, whose stationarity
   object is the thermal average of the local curvature <d^2V/da^2>.  R2B
   labels come in pairs u_i +- delta*P (P = verified K A' eigendisplacement
   of the deployed S0@450 short Hessian); the projected restoring curvature
   per pair, averaged over seeds and stripped of the q6 operator force,
   gives lambda_short(T450); sqrt(lambda*scale + Pi_K) is the absolute K
   reference.  A synthetic study on the stored R2AO forces showed the plain
   global-fit eigenvalue scatters by ~60 cm-1 across 12-config subsets and
   the force regression carries a +~30 cm-1 anharmonic bias, so neither is
   usable as the absolute estimator; the curvature estimator is exact on
   linear forces (verified to 1e-14).

2. Shape and cusp (composed TDEP fit): hiPhive effective-harmonic fit
   (cutoff 6 A) on ALL labels after force-level q6 subtraction
   (Phi_TDEP labels = F_DFT + Phi_q6 u), composed exactly like the deployed
   builder: D_ref(q) = D[Phi_short_ref](q) + Pi_full-EPC(q)|e_A'><e_A'|.
   The composition code path is replayed on the stored deployed short force
   constants and must reproduce the stored deployed frequencies bit-exactly
   before any comparison is made.

Frozen closure gates vs the deployed S0 full-EPC 450 K curve:
  * K A' absolute difference (curvature)   <= 5 cm-1
  * split-half K A' spread (6/6 seeds)     <= 5 cm-1
  * K-referenced shape RMSE / max          <= 1 / 2 cm-1
  * cusp depth relative error (d=0.003, 0.025) <= 15 %

Inputs are the label trees written by ``graphene_fd_force_convergence.py``
(``<batch dir>/<stem>_snapshots.npz`` + ``<batch dir>/labels/<stem>/``) in
the frozen phonopy supercell atom order.
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

import analyze_graphene_k_cusp_b0_dense as b0  # noqa: E402
import analyze_graphene_k_cusp_two_methods as a0  # noqa: E402
import build_graphene_fixed_smearing_thermal_full_epc as full_epc  # noqa: E402
import friedel_module as fm  # noqa: E402
import td_phonon as tdp  # noqa: E402
from phonon_accel.long_range import apply_mode_projected_correction  # noqa: E402
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402

BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
R2R = BASE / "R2R_multipolar_background"
DEPLOYED = (
    R2R
    / "R2AT_paired_full_epc_acceptance_20260827/r2ao_s0_fixed_smearing_full_epc_matrices.npz"
)
OPERATOR = ROOT / "data/graphene_r2c_eval/operators/T300_operator.npz"
BACKGROUND = ROOT / "results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
R1DIR = BASE / "R1_450k_matched_overlap_screen"
TAGGED_NPZ = R1DIR / "r2b_tagged_pairs_snapshots.npz"
OUT = BASE / "R2A_450k_dft_tdep_reference"

MASS_C_AMU = 12.0107
CUTOFF2 = 6.0
REPLAY_TOL_CM1 = 1.0e-6
CUSP_OFFSETS = (0.003, 0.025)
GATES = {
    "K_Aprime_abs_diff_cm1": 5.0,
    "split_half_K_Aprime_spread_cm1": 5.0,
    "K_referenced_shape_RMSE_cm1": 1.0,
    "K_referenced_shape_max_cm1": 2.0,
    "cusp_depth_rel_error": 0.15,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def displacement_of(positions: np.ndarray, reference: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = positions - reference
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return fractional @ cell


def collect(batch_dirs: list[Path], stems: list[str] | None) -> list[dict]:
    """Gather label records from every ``<stem>_snapshots.npz`` in the dirs."""
    records = []
    for directory in batch_dirs:
        snapshots_files = sorted(directory.glob("*_snapshots.npz"))
        if not snapshots_files:
            raise ValueError(f"no *_snapshots.npz under {directory}")
        for snapshots in snapshots_files:
            stem = snapshots.name[: -len("_snapshots.npz")]
            if stems and stem not in stems:
                continue
            with np.load(snapshots, allow_pickle=False) as data:
                sscha_indices = np.asarray(data["sscha_indices"], int)
                positions_all = np.asarray(data["positions"], float)
            for local_index, sscha_index in enumerate(sscha_indices):
                label_path = directory / "labels" / stem / f"snapshot_{local_index:03d}_k8.npz"
                with np.load(label_path, allow_pickle=False) as data:
                    forces = np.asarray(data["forces"], float)
                    energy = float(data["energy"])
                    positions = np.asarray(data["positions"], float)
                if not np.allclose(positions, positions_all[local_index], atol=1e-8):
                    raise ValueError(f"positions differ for label {sscha_index}")
                records.append(
                    {
                        "sscha_index": int(sscha_index),
                        "stem": stem,
                        "positions": positions,
                        "forces": forces,
                        "energy": energy,
                        "source": str(snapshots),
                    }
                )
    indices = [r["sscha_index"] for r in records]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate label ids across batches")
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


def compose_aprime(phonon, short_fc, qpoints, signed_q, response_cm2) -> tuple[np.ndarray, dict]:
    """A' curve of D[short_fc] + Pi_full-EPC |e_A'><e_A'| (deployed path)."""
    sequence = b0.dynamical_sequence(phonon, short_fc, qpoints)
    selected, min_overlap, max_top_difference = full_epc.track_branch(
        sequence, signed_q
    )
    applied = apply_mode_projected_correction(
        sequence["matrices"],
        sequence["scale_cm2"],
        sequence["eigenvectors"],
        selected,
        np.asarray(response_cm2, float),
    )
    diagnostics = {
        "min_branch_overlap": float(min_overlap),
        "max_top_branch_difference_cm1": float(max_top_difference),
    }
    return np.asarray(applied.tracked_frequency_cm1, float), diagnostics


def curvature_reference(delta_fc, reference, cell, tagged_labels: Path | None = None) -> tuple[dict, np.ndarray]:
    """Tagged-pair thermal curvature K reference from the R2B labels."""
    with np.load(TAGGED_NPZ, allow_pickle=False) as data:
        tagged = {key: np.asarray(data[key]) for key in data.files}
    pattern = np.asarray(tagged["k_aprime_pattern"], float)
    pattern_norm = float(tagged["pattern_norm2"].reshape(()))
    delta_a = float(np.asarray(tagged["delta_A"], float).reshape(-1)[0])
    seeds = np.asarray(tagged["seed_sscha_indices"], int)
    label_ids = np.asarray(tagged["sscha_indices"], int)
    pair_sign = np.asarray(tagged["pair_sign"], int)
    scale = float(tagged["scale_cm2"].reshape(()))
    pi_k = float(tagged["pi_K_cm2"].reshape(()))
    positions_all = np.asarray(tagged["positions"], float)

    labels = {}
    label_dir = tagged_labels or (
        TAGGED_NPZ.parent / "labels" / TAGGED_NPZ.name[: -len("_snapshots.npz")]
    )
    for local_index, label_id in enumerate(label_ids):
        label_path = label_dir / f"snapshot_{local_index:03d}_k8.npz"
        with np.load(label_path, allow_pickle=False) as data:
            positions = np.asarray(data["positions"], float)
            forces = np.asarray(data["forces"], float)
        if not np.allclose(positions, positions_all[local_index], atol=1e-8):
            raise ValueError(f"tagged label {label_id} positions differ from snapshot")
        labels[int(label_id)] = (positions, forces)

    per_seed = []
    for i, seed in enumerate(seeds):
        projected = {}
        for sign in (-1, +1):
            label_id = 1000 + 2 * i + (0 if sign < 0 else 1)
            positions, forces = labels[label_id]
            u = displacement_of(positions, reference, cell)
            forces_short = forces + np.einsum("ijab,jb->ia", delta_fc, u)
            projected[sign] = float((pattern * forces_short).sum() / pattern_norm)
        curvature = (projected[+1] - projected[-1]) / (2.0 * delta_a)
        per_seed.append(
            {
                "seed_sscha_index": int(seed),
                "curvature_eV_A2": float(curvature),
                "minus_id": 1000 + 2 * i,
                "plus_id": 1000 + 2 * i + 1,
            }
        )

    curvatures = np.array([row["curvature_eV_A2"] for row in per_seed])
    lam_ref = -float(curvatures.mean()) / MASS_C_AMU
    omega_short = float(np.sign(lam_ref) * np.sqrt(abs(lam_ref) * scale))
    omega_full = float(np.sign(lam_ref * scale + pi_k) * np.sqrt(abs(lam_ref * scale + pi_k)))
    half = len(per_seed) // 2
    lam_a = -float(curvatures[:half].mean()) / MASS_C_AMU
    lam_b = -float(curvatures[half:].mean()) / MASS_C_AMU
    omega_a = float(np.sign(lam_a * scale + pi_k) * np.sqrt(abs(lam_a * scale + pi_k)))
    omega_b = float(np.sign(lam_b * scale + pi_k) * np.sqrt(abs(lam_b * scale + pi_k)))
    summary = {
        "n_seeds": len(per_seed),
        "delta_A": delta_a,
        "lambda_short_eV_A2_amu": lam_ref,
        "omega_short_cm1": omega_short,
        "omega_full_cm1": omega_full,
        "per_seed_curvature_eV_A2": {str(r["seed_sscha_index"]): r["curvature_eV_A2"] for r in per_seed},
        "curvature_seed_std_eV_A2": float(curvatures.std()),
        "split_half_omega_full_cm1": [omega_a, omega_b],
        "split_half_spread_cm1": float(abs(omega_a - omega_b)),
        "pi_K_cm2": pi_k,
        "scale_cm2": scale,
    }
    return summary, curvatures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="append", type=Path, required=True,
                        help="batch dir containing <stem>_snapshots.npz and labels/<stem>/")
    parser.add_argument("--stems", type=str, required=True,
                        help="comma-separated snapshot stems to include (plain + r2b_tagged_pairs)")
    parser.add_argument("--tagged-labels", type=Path, default=None,
                        help="override dir for the tagged-pair labels (default: frozen R1 tree)")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    stems = args.stems.split(",")

    records = collect(args.batch, stems)
    if len(records) < 24:
        raise ValueError(f"need >= 24 labels, found {len(records)}")

    with np.load(OPERATOR, allow_pickle=False) as data:
        operator = {key: np.asarray(data[key]) for key in data.files}
    if not np.array_equal(operator["atom_mapping"], np.arange(72)):
        raise ValueError("the q6 operator does not use the identity atom mapping")
    delta_fc = np.asarray(operator["delta_fc_full"], float)
    reference = np.asarray(operator["reference_positions"], float)
    cell = np.asarray(operator["cell"], float)
    phonon, geometry_error = a0.make_phonopy(operator)
    background_positions = np.asarray(fm.load_ph(BACKGROUND).supercell.positions, float)
    operator_positions = np.asarray(phonon.supercell.positions, float)
    background_gap = float(np.abs(background_positions - operator_positions).max())
    if background_gap > 1.0e-10:
        raise ValueError(f"background/operator supercell order mismatch: {background_gap:.2e} A")
    ideal_positions = operator_positions

    with np.load(DEPLOYED, allow_pickle=False) as data:
        deployed = {key: np.asarray(data[key]) for key in data.files}
    models = [str(m) for m in deployed["short_model"]]
    temperatures = np.asarray(deployed["lattice_temperature_K"], int)
    qpoints = np.asarray(deployed["qpoints"], float)
    signed_q = np.asarray(deployed["signed_q_2pi_over_a"], float)
    response_cm2 = np.asarray(deployed["full_EPC_response_cm2"], float)
    deployed_full = np.asarray(deployed["full_EPC_frequency_cm1"], float)
    deployed_short_fc = np.asarray(deployed["short_force_constants_eV_A2"], float)
    s0_t = int(np.flatnonzero(temperatures == 450)[0])
    s0_m = int(models.index("S0"))
    r2ao_m = int(models.index("R2AO"))
    deployed_s0 = deployed_full[s0_m, s0_t]
    deployed_r2ao = deployed_full[r2ao_m, s0_t]

    # Replay validation of the composition code path on the stored matrices.
    replay_errors = {}
    for m_index, model in enumerate(models):
        for t_index, temperature in enumerate(temperatures):
            replay, _ = compose_aprime(
                phonon, deployed_short_fc[m_index, t_index], qpoints, signed_q, response_cm2
            )
            replay_errors[f"{model}_T{temperature}"] = float(
                np.max(np.abs(replay - deployed_full[m_index, t_index]))
            )
    max_replay = max(replay_errors.values())
    if max_replay > REPLAY_TOL_CM1:
        raise ValueError(
            f"composition replay mismatch vs stored deployed frequencies: {max_replay:.3e} cm-1"
        )

    # Absolute K reference from the tagged pairs.
    curvature, _ = curvature_reference(delta_fc, reference, cell, args.tagged_labels)
    omega_ref_k = curvature["omega_full_cm1"]

    # Shape/cusp reference from the composed fit on all labels (q6 removed at force level).
    for record in records:
        u = displacement_of(record["positions"], reference, cell)
        record["forces"] = record["forces"] + np.einsum("ijab,jb->ia", delta_fc, u)
    fc2_fit, fit_info = fit_fc2(phonon, ideal_positions, cell, records)
    reference_curve, fit_diag = compose_aprime(
        phonon, fc2_fit, qpoints, signed_q, response_cm2
    )

    k_index = int(np.argmin(np.abs(signed_q)))

    def cusp_depths(curve: np.ndarray) -> dict[str, float]:
        depths = {}
        for offset in CUSP_OFFSETS:
            left = np.interp(-offset, signed_q, curve)
            right = np.interp(offset, signed_q, curve)
            depths[f"d{offset}"] = float(0.5 * (left + right) - curve[k_index])
        return depths

    ref_depths = cusp_depths(reference_curve)
    dep_depths = cusp_depths(deployed_s0)

    shape_ref = reference_curve - reference_curve[k_index]
    shape_dep = deployed_s0 - deployed_s0[k_index]
    shape_delta = shape_ref - shape_dep

    metrics = {
        "n_labels": len(records),
        **fit_info,
        "K_Aprime_curvature_DFT_cm1": omega_ref_k,
        "K_Aprime_deployed_S0_cm1": float(deployed_s0[k_index]),
        "K_Aprime_deployed_R2AO_cm1": float(deployed_r2ao[k_index]),
        "K_Aprime_abs_diff_cm1": float(abs(omega_ref_k - deployed_s0[k_index])),
        "curvature_split_half_spread_cm1": curvature["split_half_spread_cm1"],
        "curvature_lambda_short_eV_A2_amu": curvature["lambda_short_eV_A2_amu"],
        "curvature_omega_short_cm1": curvature["omega_short_cm1"],
        "curvature_seed_std_eV_A2": curvature["curvature_seed_std_eV_A2"],
        "fit_curve_K_cm1": float(reference_curve[k_index]),
        "K_referenced_shape_RMSE_cm1": float(np.sqrt((shape_delta**2).mean())),
        "K_referenced_shape_max_cm1": float(np.abs(shape_delta).max()),
        "cusp_depth_fit_cm1": ref_depths,
        "cusp_depth_deployed_cm1": dep_depths,
        "cusp_depth_rel_error": {
            key: abs(ref_depths[key] - dep_depths[key]) / abs(dep_depths[key])
            for key in ref_depths
        },
        "fit_min_branch_overlap": fit_diag["min_branch_overlap"],
        "fit_max_top_branch_difference_cm1": fit_diag["max_top_branch_difference_cm1"],
    }
    gates = {
        "K_Aprime_curvature_abs_diff_le_5": metrics["K_Aprime_abs_diff_cm1"] <= GATES["K_Aprime_abs_diff_cm1"],
        "curvature_split_half_spread_le_5": metrics["curvature_split_half_spread_cm1"] <= GATES["split_half_K_Aprime_spread_cm1"],
        "shape_RMSE_le_1": metrics["K_referenced_shape_RMSE_cm1"] <= GATES["K_referenced_shape_RMSE_cm1"],
        "shape_max_le_2": metrics["K_referenced_shape_max_cm1"] <= GATES["K_referenced_shape_max_cm1"],
        "cusp_depth_le_15pct": all(
            value <= GATES["cusp_depth_rel_error"]
            for value in metrics["cusp_depth_rel_error"].values()
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "dft_reference.npz",
        fc2_fit_eV_A2=fc2_fit,
        qpoints=qpoints,
        signed_q_2pi_over_a=signed_q,
        aprime_frequency_cm1=reference_curve,
        deployed_s0_aprime_frequency_cm1=deployed_s0,
        deployed_r2ao_aprime_frequency_cm1=deployed_r2ao,
        sscha_indices=np.array(sorted(r["sscha_index"] for r in records), int),
        curvature_omega_full_cm1=np.array(omega_ref_k),
        curvature_lambda_short=np.array(curvature["lambda_short_eV_A2_amu"]),
        cutoff2=np.array(CUTOFF2),
        fit_rmse_eV_A=np.array(fit_info["fit_rmse_eV_A"]),
    )
    payload = {
        "status": "all_gates_passed" if all(gates.values()) else "gate_failure",
        "scope": "independent 450 K fixed-smearing DFT reference vs deployed S0 full-EPC curve",
        "reference_construction": {
            "absolute_K": "tagged-pair thermal curvature (SSCHA stationarity object <d2V/da2>_T), q6 operator force subtracted, shared Pi_full-EPC(K) added",
            "shape_cusp": "hiPhive effective harmonic fit (cutoff2=6.0 A) on all labels after force-level q6 subtraction, composed exactly like the deployed builder",
            "estimator_rationale": (
                "synthetic study on stored R2AO forces: global-fit K eigenvalue scatters ~60 cm-1 "
                "across 12-config subsets; A' force regression is biased ~+30 cm-1 above the SSCHA "
                "eigenvalue; tagged-pair curvature is exact on linear forces (1e-14)"
            ),
        },
        "condition": {
            "lattice_temperature_K": 450.0,
            "degauss_Ry": 0.0019000869380739254,
            "operator": str(OPERATOR),
            "operator_sha256": sha256(OPERATOR),
            "background": str(BACKGROUND),
            "background_sha256": sha256(BACKGROUND),
            "background_operator_supercell_max_gap_A": background_gap,
            "operator_geometry_error_A": float(geometry_error),
            "deployed_matrices": str(DEPLOYED),
            "deployed_matrices_sha256": sha256(DEPLOYED),
            "tagged_pairs": str(TAGGED_NPZ),
            "tagged_pairs_sha256": sha256(TAGGED_NPZ),
            "cutoff2_A": CUTOFF2,
        },
        "replay_validation_max_abs_cm1": replay_errors,
        "replay_tolerance_cm1": REPLAY_TOL_CM1,
        "curvature_reference": curvature,
        "gates_frozen": GATES,
        "metrics": metrics,
        "gates": gates,
        "batches": [str(batch) for batch in args.batch],
        "stems_included": sorted({r["stem"] for r in records}),
    }
    (args.output_dir / "acceptance.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"metrics": metrics, "gates": gates}, indent=2))
    print(f"wrote {args.output_dir / 'acceptance.json'}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
