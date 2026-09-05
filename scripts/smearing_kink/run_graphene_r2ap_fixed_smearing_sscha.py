#!/usr/bin/env python3
"""Run one formal R2AO SSCHA case at fixed electronic smearing.

The lattice temperature is 300, 450, or 600 K while the electronic operator
is fixed to ``degauss = 0.001900086938... Ry`` for every case.  The short-range
calculator is the frozen graphene foundation plus the conservative R2AO
Taylor-null spectral candidate.  The frozen q6 harmonic operator is added in
the CellConstructor atom order.

R2AO did not pass the per-configuration OOF force gate.  This runner therefore
labels every output as a spectral-sensitivity result; quantitative Kohn-cusp
and model-sensitivity gates remain mandatory before any completion claim.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for item in (HERE, ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import friedel_module as fm  # noqa: E402
import run_graphene_physical_q0_sscha as q0  # noqa: E402
import td_common as tdc  # noqa: E402
import td_phonon as tdp  # noqa: E402
from graphene_r2ao_step32_runtime import (  # noqa: E402
    R2XCorrectionCalculator,
    load_step32_spectral_checkpoint,
)
from graphene_r2r0_formal import load_endpoint  # noqa: E402
from train_graphene_r2s_conditional_mlp import recommended_inputs  # noqa: E402


FORMAT = "graphene_r2ap_fixed_smearing_r2ao_sscha_v1"
ALLOWED_TEMPERATURES = (300, 450, 600)
FIXED_OPERATOR_TEMPERATURE = 300
R2AO_CANONICAL_CELL_TOLERANCE_A = 5.0e-12


class CanonicalCellCalculator(Calculator):
    """Evaluate a frozen calculator on a byte-canonical, equivalent cell.

    CellConstructor round-trips the 6x6 cell with a roughly 1e-15 A change.
    R2AO intentionally binds its float32-origin reference graph byte-for-byte,
    so the roundoff must be removed before entering that frozen boundary.  The
    wrapper rejects any material cell change and leaves Cartesian positions,
    atom order, the foundation calculator, and the harmonic operator untouched.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base: Calculator,
        canonical_cell: np.ndarray,
        *,
        tolerance_A: float = R2AO_CANONICAL_CELL_TOLERANCE_A,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.base = base
        self.canonical_cell = np.asarray(canonical_cell, dtype=np.float64).copy()
        if self.canonical_cell.shape != (3, 3):
            raise ValueError("R2AP canonical cell must be 3x3")
        self.tolerance_A = float(tolerance_A)

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if self.atoms is None:
            raise ValueError("R2AP canonical-cell wrapper requires atoms")
        mismatch = float(
            np.max(
                np.abs(
                    np.asarray(self.atoms.cell, dtype=np.float64)
                    - self.canonical_cell
                )
            )
        )
        if mismatch > self.tolerance_A:
            raise ValueError(
                "R2AP live cell differs materially from the frozen R2AO cell"
            )
        probe = self.atoms.copy()
        probe.set_cell(self.canonical_cell, scale_atoms=False)
        self.base.calculate(probe, list(properties), all_changes)
        self.results = {
            "energy": float(self.base.results["energy"]),
            "forces": np.asarray(self.base.results["forces"], dtype=np.float64).copy(),
        }


def resolved_record(record: dict) -> Path:
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=True)
    if q0.sha256(path) != record["sha256"]:
        raise ValueError(f"R2AP frozen input hash changed: {path}")
    return path


def load_operator(record: dict) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = resolved_record(record)
    with np.load(path, allow_pickle=False) as data:
        raw_force_constants = np.asarray(data["delta_fc_full"], dtype=np.float64)
        raw_reference = np.asarray(data["reference_positions"], dtype=np.float64)
        atom_mapping = np.asarray(data["atom_mapping"], dtype=int)
        cell = np.asarray(data["cell"], dtype=np.float64)
    if sorted(atom_mapping.tolist()) != list(range(len(atom_mapping))):
        raise ValueError("R2AP operator atom mapping is not a permutation")
    return (
        path,
        raw_force_constants[atom_mapping][:, atom_mapping],
        raw_reference[atom_mapping],
        cell,
        atom_mapping,
    )


def initial_force_constants(manifest: dict, temperature: int) -> tuple[np.ndarray, dict]:
    tdep_record = manifest["initial_physical_tdep"][str(temperature)]
    tdep_path = resolved_record(tdep_record)
    with np.load(tdep_path, allow_pickle=False) as data:
        physical = np.asarray(data["pooled_fc2"], dtype=np.float64)
    _, matched_operator, _, _, _ = load_operator(
        manifest["operators"][str(temperature)]
    )
    _, fixed_operator, _, _, _ = load_operator(
        manifest["operators"][str(FIXED_OPERATOR_TEMPERATURE)]
    )
    if physical.shape != matched_operator.shape or physical.shape != fixed_operator.shape:
        raise ValueError("R2AP initial/operator force-constant shapes differ")
    return physical - matched_operator + fixed_operator, {
        "kind": "matched_TDEP_minus_matched_q6_plus_fixed_q6",
        "physical_tdep": {**tdep_record, "resolved_path": str(tdep_path)},
        "matched_operator_temperature_K": temperature,
        "fixed_operator_temperature_K": FIXED_OPERATOR_TEMPERATURE,
        "formula": "Phi_TDEP(Tlat,Tel=Tlat) - Phi_q6(Tel=Tlat) + Phi_q6(Tel=300K)",
    }


def environment_versions() -> dict[str, str]:
    output = {}
    for name in ("python-sscha", "cellconstructor", "phonopy", "mace-torch", "torch"):
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "unknown"
    return output


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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--lattice-temperature", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nsegments", type=int, default=60)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.check_environment:
        return check_environment()
    if args.manifest is None or args.lattice_temperature is None or args.output_dir is None:
        parser.error("--manifest, --lattice-temperature, and --output-dir are required")
    if args.lattice_temperature not in ALLOWED_TEMPERATURES:
        raise ValueError("R2AP lattice temperature must be 300, 450, or 600 K")

    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "R2AP_FIXED_SMEARING_SSCHA_FROZEN":
        raise ValueError("R2AP manifest is not frozen")
    if manifest["runner_sha256"] != q0.sha256(Path(__file__).resolve()):
        raise ValueError("R2AP runner source changed after manifest freeze")
    for source_record in manifest["source_files"].values():
        resolved_record(source_record)
    if manifest["fixed_operator_temperature_K"] != FIXED_OPERATOR_TEMPERATURE:
        raise ValueError("R2AP fixed electronic operator changed")
    protocol = manifest["protocol"]
    if args.lattice_temperature not in protocol["temperatures_K"]:
        raise ValueError("R2AP temperature is outside the frozen protocol")
    n_configs = int(protocol["n_configs"])
    max_populations = int(protocol["max_populations"])
    random_seed = int(protocol["random_seeds"][str(args.lattice_temperature)])

    mechanics_path = resolved_record(manifest["R2AO_mechanics_summary"])
    mechanics = json.loads(mechanics_path.read_text())
    if mechanics.get("status") != "R2AO_SPECTRAL_CANDIDATE_MECHANICS_PASSED":
        raise ValueError("R2AP requires passed R2AO mechanics")
    candidate_summary_path = resolved_record(manifest["R2AO_candidate_summary"])
    candidate_summary = json.loads(candidate_summary_path.read_text())
    if candidate_summary.get("force_gate_approved") is not False:
        raise ValueError("R2AP candidate force-gate boundary changed")

    output_dir = args.output_dir.resolve()
    acceptance_path = output_dir / "acceptance.json"
    result_path = output_dir / "result.npz"
    if acceptance_path.is_file() and result_path.is_file():
        existing = json.loads(acceptance_path.read_text())
        if (
            existing.get("format") == FORMAT
            and existing.get("lattice_temperature_K") == args.lattice_temperature
            and existing.get("result_sha256") == q0.sha256(result_path)
        ):
            print(json.dumps({"status": "reused_terminal_case", **existing}, indent=2))
            return 0 if existing.get("status") == "passed" else 1

    import torch
    from ase.calculators.mixing import SumCalculator
    from friedel_calc import FixedHarmonicCorrection, FriedelMACECalculator
    from phonon_accel.phonons import phonopy_to_ase
    from sscha.Ensemble import Ensemble
    from sscha.Relax import SSCHA
    from sscha.SchaMinimizer import SSCHA_Minimizer

    if not hasattr(np, "int"):
        np.int = int
        np.float = float
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    background_path = resolved_record(manifest["background"])
    phonon = fm.load_ph(background_path)
    initial_fc2, initial_record = initial_force_constants(
        manifest, args.lattice_temperature
    )
    (
        dynamical,
        dimension,
        phonopy_for_cc,
        cc_for_phonopy,
        cc_reference,
    ) = q0.phonopy_fc_to_cc_dyn(phonon, initial_fc2)
    dynamical.ForcePositiveDefinite()
    dynamical.Symmetrize()

    foundation_path = resolved_record(manifest["foundation_model"])
    foundation = tdc.get_mace_calc(str(foundation_path), device=args.device)
    endpoint = load_endpoint(recommended_inputs(), args.device)
    for parameter in endpoint.parameters():
        parameter.requires_grad_(False)
    endpoint.eval()
    checkpoint_path = resolved_record(manifest["R2AO_checkpoint"])
    checkpoint = load_step32_spectral_checkpoint(
        checkpoint_path, expected_sha256=manifest["R2AO_checkpoint"]["sha256"]
    )
    reference6 = __import__("ase.io", fromlist=["read"]).read(
        recommended_inputs().reference_6x6, index=0
    )
    r2ao_raw = R2XCorrectionCalculator(
        endpoint, reference6, checkpoint, device=args.device
    )
    r2ao = CanonicalCellCalculator(
        r2ao_raw,
        np.asarray(reference6.cell, dtype=np.float64),
    )
    short_calculator = SumCalculator([foundation, r2ao])

    operator_record = manifest["operators"][str(FIXED_OPERATOR_TEMPERATURE)]
    (
        operator_path,
        operator_phonopy,
        operator_reference,
        operator_cell,
        operator_atom_mapping,
    ) = load_operator(operator_record)
    phonopy_reference = phonopy_to_ase(phonon.supercell)
    if not np.allclose(
        np.asarray(phonopy_reference.cell), operator_cell, atol=2.0e-5, rtol=0.0
    ):
        raise ValueError("R2AP operator and phonopy supercells differ")
    operator_reference_mismatch = q0.periodic_reference_max_distance(
        operator_reference, np.asarray(phonopy_reference.positions), operator_cell
    )
    if operator_reference_mismatch > 2.0e-5:
        raise ValueError("R2AP operator reference differs from phonopy reference")
    cc_reference_mismatch = q0.periodic_reference_max_distance(
        np.asarray(phonopy_reference.positions)[phonopy_for_cc],
        np.asarray(cc_reference.positions),
        operator_cell,
    )
    if cc_reference_mismatch > 2.0e-5:
        raise ValueError("R2AP CellConstructor reference mapping differs")
    r2ao_reference_cell_mismatch = float(
        np.max(
            np.abs(
                np.asarray(cc_reference.cell, dtype=np.float64)
                - np.asarray(reference6.cell, dtype=np.float64)
            )
        )
    )
    if r2ao_reference_cell_mismatch > R2AO_CANONICAL_CELL_TOLERANCE_A:
        raise ValueError("R2AP CellConstructor cell differs materially from R2AO")
    operator_cc = operator_phonopy[phonopy_for_cc][:, phonopy_for_cc]
    correction = FixedHarmonicCorrection(operator_cc)
    calculator = FriedelMACECalculator(
        short_calculator,
        cc_reference,
        correction,
        FIXED_OPERATOR_TEMPERATURE,
    )

    if args.preflight_only:
        probe = cc_reference.copy()
        probe.positions[0, 0] += 1.0e-3
        probe.calc = calculator
        energy = float(probe.get_potential_energy())
        force = np.asarray(probe.get_forces(), dtype=np.float64)
        if not np.isfinite(energy) or not np.all(np.isfinite(force)):
            raise ValueError("R2AP preflight calculator produced a non-finite value")
        print(
            json.dumps(
                {
                    "status": "R2AP_PREFLIGHT_PASSED",
                    "lattice_temperature_K": args.lattice_temperature,
                    "fixed_smearing_degauss_Ry": operator_record["degauss_Ry"],
                    "probe_energy_eV": energy,
                    "probe_force_max_abs_eV_A": float(np.max(np.abs(force))),
                    "R2AO_CellConstructor_cell_mismatch_A": r2ao_reference_cell_mismatch,
                    "R2AO_cell_canonicalization_tolerance_A": R2AO_CANONICAL_CELL_TOLERANCE_A,
                    "device": args.device,
                },
                indent=2,
            )
        )
        return 0

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
            N_configs=n_configs,
            max_pop=max_populations,
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
    supercell_frequencies_cm = np.asarray(supercell_frequencies, dtype=float) * q0.RY_TO_CM
    free_energy_fc2 = q0.cc_dyn_to_full_fc(hessian, dimension, cc_for_phonopy)
    distance, bands_thz, label_positions, labels = tdp.band_from_phonopy(
        phonon, free_energy_fc2, npoints=3 * args.nsegments + 1
    )
    bands = np.asarray(bands_thz, dtype=float) * 33.35641
    finite = bool(
        np.isfinite(supercell_frequencies_cm).all()
        and np.isfinite(bands).all()
        and np.isfinite(free_energy_fc2).all()
    )
    imaginary_count = int(np.sum(supercell_frequencies_cm < -1.0))
    passed = bool(converged and finite and imaginary_count == 0)
    q0.atomic_npz(
        result_path,
        format=np.array(FORMAT),
        lattice_temperature_K=np.array(args.lattice_temperature),
        fixed_operator_temperature_K=np.array(FIXED_OPERATOR_TEMPERATURE),
        fixed_smearing_degauss_Ry=np.array(operator_record["degauss_Ry"]),
        distance=distance,
        label_positions=label_positions,
        labels=labels,
        frequency_cm_1=bands,
        supercell_frequency_cm_1=supercell_frequencies_cm,
        free_energy_fc2_eV_A2=free_energy_fc2,
        converged=np.array(converged),
        final_population=np.array(final_population),
        n_configs=np.array(n_configs),
        max_populations=np.array(max_populations),
        random_seed=np.array(random_seed),
        operator_atom_mapping=operator_atom_mapping,
        phonopy_for_CellConstructor_atom_mapping=phonopy_for_cc,
        CellConstructor_for_phonopy_atom_mapping=cc_for_phonopy,
        operator_reference_mismatch_A=np.array(operator_reference_mismatch),
        CellConstructor_reference_mismatch_A=np.array(cc_reference_mismatch),
        R2AO_CellConstructor_cell_mismatch_A=np.array(
            r2ao_reference_cell_mismatch
        ),
        R2AO_cell_canonicalization_tolerance_A=np.array(
            R2AO_CANONICAL_CELL_TOLERANCE_A
        ),
    )
    acceptance = {
        "format": FORMAT,
        "status": "passed" if passed else "not_converged_or_unstable",
        "spectral_sensitivity_only": True,
        "force_gate_approved": False,
        "lattice_temperature_K": args.lattice_temperature,
        "fixed_smearing_degauss_Ry": operator_record["degauss_Ry"],
        "electronic_operator_temperature_label_K": FIXED_OPERATOR_TEMPERATURE,
        "converged": converged,
        "all_outputs_finite": finite,
        "final_population": final_population,
        "maximum_populations": max_populations,
        "n_configs_per_population": n_configs,
        "random_seed": random_seed,
        "supercell_min_frequency_cm-1": float(np.min(supercell_frequencies_cm)),
        "supercell_imaginary_modes_below_minus_1_cm-1": imaginary_count,
        "R2AO_CellConstructor_cell_mismatch_A": r2ao_reference_cell_mismatch,
        "R2AO_cell_canonicalization_tolerance_A": R2AO_CANONICAL_CELL_TOLERANCE_A,
        "result": str(result_path),
        "result_sha256": q0.sha256(result_path),
        "input_provenance": {
            "manifest": {"path": str(manifest_path), "sha256": q0.sha256(manifest_path)},
            "initial_hessian": initial_record,
            "foundation_model": manifest["foundation_model"],
            "R2AO_checkpoint": manifest["R2AO_checkpoint"],
            "R2AO_candidate_summary": manifest["R2AO_candidate_summary"],
            "R2AO_mechanics_summary": manifest["R2AO_mechanics_summary"],
            "background": manifest["background"],
            "fixed_operator": {**operator_record, "resolved_path": str(operator_path)},
        },
        "versions": environment_versions(),
        "next_stage": (
            "full-EPC dense K-line reconstruction and quantitative cusp/model-sensitivity gates"
            if passed
            else "stop this temperature; inspect SSCHA convergence/stability"
        ),
    }
    q0.atomic_json(acceptance_path, acceptance)
    print(json.dumps(acceptance, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
