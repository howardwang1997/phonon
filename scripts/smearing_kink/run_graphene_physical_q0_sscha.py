#!/usr/bin/env python3
"""Run one frozen graphene Q0/X0 SSCHA condition and write a convergence gate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


RY_TO_CM = 109736.75
EV_A2_TO_RY_BOHR2 = (1.0 / 13.605693009) / (1.8897259886**2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def hexagonal_qpath(nsegment: int = 60):
    """Return a duplicate-free Gamma-M-K-Gamma path including all endpoints."""
    if nsegment < 2:
        raise ValueError("nsegment must be at least two")
    points = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0], [0.0, 0.0, 0.0]],
        float,
    )
    labels = np.array([r"$\Gamma$", "M", "K", r"$\Gamma$"])
    qpoints = []
    label_indices = [0]
    for segment in range(len(points) - 1):
        values = np.linspace(points[segment], points[segment + 1], nsegment + 1)
        if segment:
            values = values[1:]
        qpoints.extend(values)
        label_indices.append(len(qpoints) - 1)
    qpoints = np.asarray(qpoints, float)
    increments = np.linalg.norm(np.diff(qpoints, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(increments)])
    label_positions = distance[np.asarray(label_indices, int)]
    return qpoints, distance, label_positions, labels


def full_fc_to_matrix(force_constants: np.ndarray) -> np.ndarray:
    force_constants = np.asarray(force_constants, float)
    if (
        force_constants.ndim != 4
        or force_constants.shape[0] != force_constants.shape[1]
        or force_constants.shape[2:] != (3, 3)
    ):
        raise ValueError("force constants must have shape (N,N,3,3)")
    return force_constants.transpose(0, 2, 1, 3).reshape(
        3 * force_constants.shape[0], 3 * force_constants.shape[0]
    )


def matrix_to_full_fc(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 3:
        raise ValueError("force-constant matrix must be square with dimension 3N")
    natoms = matrix.shape[0] // 3
    return matrix.reshape(natoms, 3, natoms, 3).transpose(0, 2, 1, 3)


def supercell_diagonal(phonon) -> np.ndarray:
    matrix = np.asarray(phonon.supercell_matrix, int)
    diagonal = np.diag(np.diag(matrix))
    if not np.array_equal(matrix, diagonal) or np.any(np.diag(matrix) <= 0):
        raise ValueError("Q0 currently requires a positive diagonal supercell matrix")
    return np.asarray(np.diag(matrix), dtype=np.intc)


def phonopy_cc_order_mapping(phonon, cc_super_structure) -> tuple[np.ndarray, np.ndarray]:
    """Return phonopy-for-CC and inverse atom maps using periodic coordinates."""
    from phonon_accel.phonons import phonopy_to_ase

    phonopy_atoms = phonopy_to_ase(phonon.supercell)
    cc_atoms = cc_super_structure.get_ase_atoms()
    if len(phonopy_atoms) != len(cc_atoms):
        raise ValueError("phonopy and CellConstructor supercells have different sizes")
    if not np.allclose(
        np.asarray(phonopy_atoms.cell), np.asarray(cc_atoms.cell), atol=2.0e-5, rtol=0.0
    ):
        raise ValueError("phonopy and CellConstructor supercell vectors differ")
    cell = np.asarray(phonopy_atoms.cell, float)
    phonopy_scaled = np.asarray(phonopy_atoms.get_scaled_positions(wrap=True), float)
    cc_scaled = np.asarray(cc_atoms.get_scaled_positions(wrap=True), float)
    phonopy_numbers = np.asarray(phonopy_atoms.numbers, int)
    cc_numbers = np.asarray(cc_atoms.numbers, int)
    phonopy_for_cc = np.full(len(cc_atoms), -1, int)
    unused = set(range(len(phonopy_atoms)))
    for cc_index, (scaled, number) in enumerate(zip(cc_scaled, cc_numbers)):
        candidates = np.array(
            [index for index in unused if phonopy_numbers[index] == number], int
        )
        if not len(candidates):
            raise ValueError("could not match a CellConstructor atom by element")
        fractional = phonopy_scaled[candidates] - scaled
        fractional -= np.round(fractional)
        distances = np.linalg.norm(fractional @ cell, axis=1)
        best = int(np.argmin(distances))
        if float(distances[best]) > 2.0e-5:
            raise ValueError(
                "could not match phonopy and CellConstructor atoms within tolerance"
            )
        phonopy_index = int(candidates[best])
        phonopy_for_cc[cc_index] = phonopy_index
        unused.remove(phonopy_index)
    if unused or sorted(phonopy_for_cc.tolist()) != list(range(len(phonopy_atoms))):
        raise ValueError("phonopy/CellConstructor atom mapping is not a permutation")
    cc_for_phonopy = np.argsort(phonopy_for_cc)
    return phonopy_for_cc, cc_for_phonopy


def phonopy_fc_to_cc_dyn(phonon, force_constants: np.ndarray):
    import cellconstructor as CC
    import cellconstructor.ForceTensor  # noqa: F401
    import cellconstructor.Structure  # noqa: F401
    from ase import Atoms

    natoms = len(phonon.supercell)
    force_constants = np.asarray(force_constants, float)
    if force_constants.shape != (natoms, natoms, 3, 3):
        raise ValueError(
            f"expected full FC2 shape {(natoms, natoms, 3, 3)}, got {force_constants.shape}"
        )
    primitive = phonon.primitive
    unitcell = Atoms(
        numbers=primitive.numbers,
        scaled_positions=primitive.scaled_positions,
        cell=primitive.cell,
        pbc=True,
    )
    structure = CC.Structure.Structure()
    structure.generate_from_ase_atoms(unitcell)
    dimension = supercell_diagonal(phonon)
    super_structure = structure.generate_supercell(dimension)
    phonopy_for_cc, cc_for_phonopy = phonopy_cc_order_mapping(
        phonon, super_structure
    )
    tensor = CC.ForceTensor.Tensor2(structure, super_structure, dimension)
    force_constants_cc = force_constants[phonopy_for_cc][:, phonopy_for_cc]
    matrix = full_fc_to_matrix(force_constants_cc) * EV_A2_TO_RY_BOHR2
    tensor.SetupFromTensor(np.asarray(matrix, dtype="double", order="F"))
    return (
        tensor.GeneratePhonons(dimension),
        dimension,
        phonopy_for_cc,
        cc_for_phonopy,
        super_structure.get_ase_atoms(),
    )


def cc_dyn_to_full_fc(
    dynamical, dimension: np.ndarray, cc_for_phonopy: np.ndarray
) -> np.ndarray:
    matrix_complex = np.asarray(
        dynamical.GetRealSpaceFC(tuple(int(value) for value in dimension))
    )
    imaginary_max = float(np.max(np.abs(np.imag(matrix_complex))))
    if imaginary_max > 1.0e-10:
        raise ValueError(
            f"CellConstructor real-space FC2 has imaginary residue {imaginary_max:g}"
        )
    matrix_ry_bohr2 = np.real(matrix_complex)
    force_constants_cc = matrix_to_full_fc(matrix_ry_bohr2 / EV_A2_TO_RY_BOHR2)
    return force_constants_cc[cc_for_phonopy][:, cc_for_phonopy]


def scalar(data, key: str):
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} is not scalar")
    return value.reshape(()).item()


def verified_path(record: dict) -> Path:
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"frozen input hash changed: {path}")
    return path


def operator_fc(
    freeze: dict, temperature: int
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record = freeze["source_L0"]["operators"][str(temperature)]
    path = verified_path(record)
    with np.load(path, allow_pickle=False) as data:
        raw_force_constants = np.asarray(data["delta_fc_full"], float)
        raw_reference = np.asarray(data["reference_positions"], float)
        atom_mapping = np.asarray(data["atom_mapping"], int)
        cell = np.asarray(data["cell"], float)
    if sorted(atom_mapping.tolist()) != list(range(len(atom_mapping))):
        raise ValueError("operator atom_mapping is not a permutation")
    force_constants = raw_force_constants[atom_mapping][:, atom_mapping]
    reference = raw_reference[atom_mapping]
    return path, force_constants, reference, cell, atom_mapping


def periodic_reference_max_distance(
    left: np.ndarray, right: np.ndarray, cell: np.ndarray
) -> float:
    if np.asarray(left).shape != np.asarray(right).shape:
        raise ValueError("reference arrays have different shapes")
    displacement = np.asarray(left, float) - np.asarray(right, float)
    fractional = displacement @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return float(np.max(np.linalg.norm(fractional @ cell, axis=1)))


def protocol_for(freeze: dict, phase: str, lattice_temperature: int, operator_temperature: int):
    q0 = freeze["Q0_protocol"]
    if phase == "benchmark":
        protocol = q0["benchmark"]
        expected = (
            int(protocol["lattice_temperature_K"]),
            int(protocol["operator_temperature_K"]),
        )
    elif phase == "formal":
        protocol = q0["formal"]
        if lattice_temperature not in protocol["temperatures_K"]:
            raise ValueError("formal Q0 temperature is outside the frozen list")
        expected = (lattice_temperature, lattice_temperature)
        protocol = {
            "n_configs": protocol["n_configs"],
            "max_populations": protocol["max_populations"],
            "random_seed": protocol["random_seeds"][str(lattice_temperature)],
        }
    elif phase == "x0_first":
        protocol = freeze["X0_protocol"]["first_non_diagonal_condition"]
        expected = (
            int(protocol["lattice_temperature_K"]),
            int(protocol["operator_temperature_K"]),
        )
    else:
        raise ValueError(f"unsupported phase: {phase}")
    if expected != (lattice_temperature, operator_temperature):
        raise ValueError(
            f"condition {(lattice_temperature, operator_temperature)} differs from frozen {expected}"
        )
    return {
        "n_configs": int(protocol["n_configs"]),
        "max_populations": int(protocol["max_populations"]),
        "random_seed": int(protocol["random_seed"]),
    }


def initial_force_constants(
    freeze: dict, phase: str, lattice_temperature: int, operator_temperature: int
) -> tuple[np.ndarray, dict]:
    if phase != "x0_first":
        record = freeze["source_L0"]["results"][str(lattice_temperature)]["physical_tdep"]
        path = verified_path(record)
        with np.load(path, allow_pickle=False) as data:
            force_constants = np.asarray(data["pooled_fc2"], float)
        return force_constants, {"kind": "L0_pooled_physical_TDEP", **record}

    q0_root = Path(freeze["paths"]["q0_root"])
    q0_path = q0_root / f"formal_T{lattice_temperature}" / "result.npz"
    q0_acceptance = q0_root / f"formal_T{lattice_temperature}" / "acceptance.json"
    acceptance = json.loads(q0_acceptance.read_text())
    if acceptance.get("status") != "passed" or acceptance.get("converged") is not True:
        raise ValueError("the diagonal Q0 source did not pass")
    if acceptance.get("result_sha256") != sha256(q0_path):
        raise ValueError("the diagonal Q0 result hash differs from its acceptance record")
    with np.load(q0_path, allow_pickle=False) as data:
        diagonal = np.asarray(data["free_energy_fc2_eV_A2"], float)
    _, diagonal_operator, _, _, _ = operator_fc(freeze, lattice_temperature)
    _, target_operator, _, _, _ = operator_fc(freeze, operator_temperature)
    reconstructed = diagonal - diagonal_operator + target_operator
    return reconstructed, {
        "kind": "Q0_formula_reconstructed_trial_hessian",
        "diagonal_Q0_result": str(q0_path),
        "diagonal_Q0_result_sha256": sha256(q0_path),
        "formula": "Phi_Q0(Tlat,Tlat) - Phi_operator(Tlat) + Phi_operator(Tel)",
    }


def environment_versions() -> dict:
    versions = {}
    for distribution in ("python-sscha", "cellconstructor", "phonopy", "mace-torch", "torch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unknown"
    return versions


def check_environment() -> int:
    import cellconstructor.Phonons  # noqa: F401
    import sscha  # noqa: F401
    from sscha.Ensemble import Ensemble  # noqa: F401
    from sscha.Relax import SSCHA  # noqa: F401
    from sscha.SchaMinimizer import SSCHA_Minimizer  # noqa: F401

    print(json.dumps({"status": "environment_ok", "versions": environment_versions()}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--phase", choices=("benchmark", "formal", "x0_first"))
    parser.add_argument("--lattice-temperature", type=int)
    parser.add_argument("--operator-temperature", type=int)
    parser.add_argument("--n-configs", type=int)
    parser.add_argument("--max-populations", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--nsegments", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.check_environment:
        return check_environment()
    required = (
        "freeze_manifest",
        "phase",
        "lattice_temperature",
        "operator_temperature",
        "n_configs",
        "max_populations",
        "random_seed",
        "output_dir",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    freeze = json.loads(args.freeze_manifest.read_text())
    if freeze.get("status") != "frozen_for_Q0_X0_development":
        raise ValueError("invalid Q0/X0 freeze manifest")
    if freeze.get("locked_validation_temperatures_K_not_accessed") != [375, 525]:
        raise ValueError("locked validation boundary is missing")
    if sha256(Path(__file__).resolve()) != freeze["scripts"]["q0_sscha"]["sha256"]:
        raise ValueError("Q0 SSCHA script changed after the development protocol was frozen")
    protocol = protocol_for(
        freeze, args.phase, args.lattice_temperature, args.operator_temperature
    )
    requested = {
        "n_configs": args.n_configs,
        "max_populations": args.max_populations,
        "random_seed": args.random_seed,
    }
    if requested != protocol:
        raise ValueError(f"requested SSCHA settings {requested} differ from frozen {protocol}")

    output_dir = args.output_dir.resolve()
    existing_acceptance_path = output_dir / "acceptance.json"
    existing_result_path = output_dir / "result.npz"
    if existing_acceptance_path.is_file() and existing_result_path.is_file():
        existing = json.loads(existing_acceptance_path.read_text())
        condition = existing.get("condition", {})
        same_condition = (
            existing.get("phase") == args.phase
            and condition.get("lattice_temperature_K") == args.lattice_temperature
            and condition.get("operator_temperature_K") == args.operator_temperature
            and existing.get("n_configs_per_population") == args.n_configs
            and existing.get("maximum_populations") == args.max_populations
            and existing.get("random_seed") == args.random_seed
        )
        intact = existing.get("result_sha256") == sha256(existing_result_path)
        if same_condition and intact:
            print(json.dumps({"status": "reused_terminal_case", **existing}, indent=2))
            return 0 if existing.get("status") == "passed" else 1

    import torch
    import friedel_module as fm
    import td_common as tdc
    import td_phonon as tdp
    from ase.calculators.mixing import SumCalculator
    from friedel_calc import FixedHarmonicCorrection, FriedelMACECalculator
    from phonon_accel.phonons import phonopy_to_ase
    from sscha.Ensemble import Ensemble
    from sscha.Relax import SSCHA
    from sscha.SchaMinimizer import SSCHA_Minimizer

    if not hasattr(np, "int"):
        np.int = int
        np.float = float
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    background_record = freeze["source_L0"]["background"]
    background_path = verified_path(background_record)
    phonon = fm.load_ph(background_path)
    initial_fc2, initial_record = initial_force_constants(
        freeze, args.phase, args.lattice_temperature, args.operator_temperature
    )
    (
        dynamical,
        dimension,
        phonopy_for_cc,
        cc_for_phonopy,
        cc_reference,
    ) = phonopy_fc_to_cc_dyn(phonon, initial_fc2)
    dynamical.ForcePositiveDefinite()
    dynamical.Symmetrize()

    base_record = freeze["source_L0"]["base_model"]
    delta_record = freeze["source_L0"]["delta_model"]
    base_path = verified_path(base_record)
    delta_path = verified_path(delta_record)
    base = tdc.get_mace_calc(str(base_path), device=args.device)
    delta = tdc.get_mace_calc(str(delta_path), device=args.device)
    short_calculator = SumCalculator([base, delta])
    (
        operator_path,
        operator_phonopy,
        operator_reference,
        operator_cell,
        operator_atom_mapping,
    ) = operator_fc(freeze, args.operator_temperature)
    phonopy_reference = phonopy_to_ase(phonon.supercell)
    if not np.allclose(
        np.asarray(phonopy_reference.cell), operator_cell, atol=2.0e-5, rtol=0.0
    ):
        raise ValueError("operator and phonopy supercells differ")
    operator_reference_mismatch = periodic_reference_max_distance(
        operator_reference, np.asarray(phonopy_reference.positions), operator_cell
    )
    if operator_reference_mismatch > 2.0e-5:
        raise ValueError(
            "mapped operator reference differs from the phonopy supercell by "
            f"{operator_reference_mismatch:g} angstrom"
        )
    cc_reference_mismatch = periodic_reference_max_distance(
        np.asarray(phonopy_reference.positions)[phonopy_for_cc],
        np.asarray(cc_reference.positions),
        operator_cell,
    )
    if cc_reference_mismatch > 2.0e-5:
        raise ValueError(
            "phonopy-to-CellConstructor reference mapping differs by "
            f"{cc_reference_mismatch:g} angstrom"
        )
    operator_cc = operator_phonopy[phonopy_for_cc][:, phonopy_for_cc]
    correction = FixedHarmonicCorrection(operator_cc)
    calculator = FriedelMACECalculator(
        short_calculator, cc_reference, correction, args.operator_temperature
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    converged = False
    try:
        os.chdir(output_dir)
        ensemble = Ensemble(
            dynamical,
            T0=float(args.lattice_temperature),
            supercell=dynamical.GetSupercell(),
        )
        minimizer = SSCHA_Minimizer(ensemble)
        minimizer.min_step_dyn = 0.05
        minimizer.kong_liu_ratio = 0.5
        relaxation = SSCHA(
            minimizer,
            ase_calculator=calculator,
            N_configs=args.n_configs,
            max_pop=args.max_populations,
            save_ensemble=True,
        )
        converged = bool(
            relaxation.relax(
                get_stress=False,
                ensemble_loc=str(output_dir / "ensembles"),
            )
        )
        final_population = max(0, int(relaxation.start_pop) - 1)
        hessian = relaxation.minim.ensemble.get_free_energy_hessian()
        hessian.save_qe(str(output_dir / "free_energy_hessian_"))
    finally:
        os.chdir(previous_cwd)

    supercell_frequencies, _ = hessian.DiagonalizeSupercell()
    supercell_frequencies_cm = np.asarray(supercell_frequencies, float) * RY_TO_CM
    free_energy_fc2 = cc_dyn_to_full_fc(hessian, dimension, cc_for_phonopy)
    distance, bands_thz, label_positions, labels = tdp.band_from_phonopy(
        phonon, free_energy_fc2, npoints=3 * args.nsegments + 1
    )
    bands = np.asarray(bands_thz, float) * 33.35641
    finite = bool(
        np.isfinite(supercell_frequencies_cm).all()
        and np.isfinite(bands).all()
        and np.isfinite(free_energy_fc2).all()
    )
    passed = bool(converged and finite)
    result_path = output_dir / "result.npz"
    atomic_npz(
        result_path,
        phase=np.array(args.phase),
        lattice_temperature_K=np.array(args.lattice_temperature),
        operator_temperature_K=np.array(args.operator_temperature),
        distance=distance,
        label_positions=label_positions,
        labels=labels,
        frequency_cm_1=bands,
        supercell_frequency_cm_1=supercell_frequencies_cm,
        free_energy_fc2_eV_A2=free_energy_fc2,
        converged=np.array(converged),
        final_population=np.array(final_population),
        n_configs=np.array(args.n_configs),
        max_populations=np.array(args.max_populations),
        random_seed=np.array(args.random_seed),
        operator_atom_mapping=operator_atom_mapping,
        phonopy_for_CellConstructor_atom_mapping=phonopy_for_cc,
        CellConstructor_for_phonopy_atom_mapping=cc_for_phonopy,
        operator_reference_mismatch_A=np.array(operator_reference_mismatch),
        CellConstructor_reference_mismatch_A=np.array(cc_reference_mismatch),
    )
    acceptance = {
        "status": "passed" if passed else "not_converged",
        "phase": args.phase,
        "condition": {
            "lattice_temperature_K": args.lattice_temperature,
            "operator_temperature_K": args.operator_temperature,
            "operator_degauss_Ry": freeze["source_L0"]["operators"][
                str(args.operator_temperature)
            ]["degauss_Ry"],
            "operator_degauss_formula": "k_B*T_operator/Ry",
        },
        "converged": converged,
        "all_outputs_finite": finite,
        "final_population": final_population,
        "maximum_populations": args.max_populations,
        "n_configs_per_population": args.n_configs,
        "maximum_population_is_not_automatically_accepted": True,
        "random_seed": args.random_seed,
        "atom_order_interface": {
            "status": "validated_before_SSCHA",
            "operator_atom_mapping_applied": operator_atom_mapping.tolist(),
            "phonopy_for_CellConstructor": phonopy_for_cc.tolist(),
            "CellConstructor_for_phonopy": cc_for_phonopy.tolist(),
            "operator_reference_mismatch_A": operator_reference_mismatch,
            "CellConstructor_reference_mismatch_A": cc_reference_mismatch,
            "calculator_reference_order": "CellConstructor",
            "calculator_operator_order": "CellConstructor",
        },
        "supercell_min_frequency_cm-1": float(np.min(supercell_frequencies_cm)),
        "supercell_imaginary_modes_below_minus_1_cm-1": int(
            np.sum(supercell_frequencies_cm < -1.0)
        ),
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "input_provenance": {
            "freeze_manifest": {
                "path": str(args.freeze_manifest),
                "sha256": sha256(args.freeze_manifest),
            },
            "initial_hessian": initial_record,
            "base_model": base_record,
            "delta_model": delta_record,
            "background": background_record,
            "operator": {
                "path": str(operator_path),
                "sha256": sha256(operator_path),
            },
        },
        "versions": environment_versions(),
        "next_stage": (
            "continue frozen Q0/X0 queue"
            if passed
            else "stop; maximum population was reached without numerical convergence"
        ),
    }
    atomic_json(output_dir / "acceptance.json", acceptance)
    print(json.dumps(acceptance, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
