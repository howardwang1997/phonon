#!/usr/bin/env python3
"""Block-bootstrap the frozen 450 K sampling gate before DFT force labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
from hiphive.utilities import prepare_structures

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
import td_phonon as tdp  # noqa: E402
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402
from recompute_graphene_tdep_from_checkpoints import subtract_operator  # noqa: E402


CM_PER_THz = 33.35641
PAIR_LABELS = ("0-1", "0-2", "1-2")
OBSERVABLE_LABELS = (
    "instantaneous_temperature_K",
    "potential_energy_eV_atom",
    "force_rms_eV_A",
    "displacement_rms_A",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labelled_path(value: str) -> tuple[int, Path]:
    try:
        seed, raw_path = value.split("=", 1)
        return int(seed), Path(raw_path)
    except ValueError as error:
        raise argparse.ArgumentTypeError("inputs must use SEED=/path/to/file") from error


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


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    """Initial-positive-sequence estimate in units of saved snapshots."""
    centered = np.asarray(values, float) - float(np.mean(values))
    variance = float(np.dot(centered, centered) / len(centered))
    if not np.isfinite(variance) or variance <= np.finfo(float).eps:
        return 0.5
    tau = 0.5
    for lag in range(1, len(centered)):
        covariance = float(np.dot(centered[:-lag], centered[lag:]) / (len(centered) - lag))
        correlation = covariance / variance
        if not np.isfinite(correlation) or correlation <= 0.0:
            break
        tau += correlation
    return float(tau)


def moving_block_indices(n_items: int, block_length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n_items / block_length)
    starts = rng.integers(0, n_items, size=n_blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n_items).reshape(-1)[:n_items]


def solve_parameters(gram_blocks: np.ndarray, rhs_blocks: np.ndarray, indices: np.ndarray) -> np.ndarray:
    gram = np.sum(gram_blocks[indices], axis=0)
    rhs = np.sum(rhs_blocks[indices], axis=0)
    scale = np.sqrt(np.maximum(np.diag(gram), np.finfo(float).tiny))
    inverse_scale = 1.0 / scale
    scaled_gram = gram * inverse_scale[:, None] * inverse_scale[None, :]
    scaled_rhs = rhs * inverse_scale
    scaled_parameters = np.linalg.lstsq(scaled_gram, scaled_rhs, rcond=-1)[0]
    return scaled_parameters * inverse_scale


def fc_basis(cluster_space, ideal) -> np.ndarray:
    n_parameters = int(cluster_space.n_dofs)
    basis = []
    for index in range(n_parameters):
        parameters = np.zeros(n_parameters)
        parameters[index] = 1.0
        potential = ForceConstantPotential(cluster_space, parameters)
        force_constants = potential.get_force_constants(ideal).get_fc_array(order=2)
        basis.append(np.asarray(force_constants, float))
    return np.asarray(basis)


def spectrum(
    phonon,
    basis: np.ndarray,
    parameters: np.ndarray,
    npoints: int,
    additive_force_constants: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    force_constants = np.tensordot(parameters, basis, axes=(0, 0))
    if additive_force_constants is not None:
        force_constants = force_constants + additive_force_constants
    distance, frequency, _, _ = tdp.band_from_phonopy(
        phonon, force_constants, npoints=npoints
    )
    return distance, np.asarray(frequency, float) * CM_PER_THz


def full_band_mae(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"spectrum shapes differ: {left.shape} != {right.shape}")
    return float(np.mean(np.abs(left - right)))


def load_reference_spectrum(path: Path, temperature: int) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"T{temperature}"
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data[f"{prefix}_dist"], float),
            np.asarray(data[f"{prefix}_freq"], float) * CM_PER_THz,
        )


def percentile_higher(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile, method="higher"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--checkpoint", type=labelled_path, action="append", required=True)
    parser.add_argument("--seed-tdep", type=labelled_path, action="append", required=True)
    manifest_group = parser.add_mutually_exclusive_group(required=True)
    manifest_group.add_argument("--freeze-manifest", type=Path)
    manifest_group.add_argument("--l0-freeze-manifest", type=Path)
    parser.add_argument("--sampling-acceptance", type=Path, required=True)
    parser.add_argument("--temperature", type=int, default=450)
    parser.add_argument("--cutoff2", type=float, default=6.0)
    parser.add_argument("--npoints", type=int, default=201)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=450001)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.replicates <= 0 or args.checkpoint_every <= 0:
        parser.error("replicates and checkpoint-every must be positive")
    checkpoint_paths = dict(args.checkpoint)
    reference_paths = dict(args.seed_tdep)
    if set(checkpoint_paths) != {0, 1, 2} or set(reference_paths) != {0, 1, 2}:
        parser.error("exactly seeds 0, 1, and 2 are required for checkpoints and TDEP files")

    manifest_path = args.l0_freeze_manifest or args.freeze_manifest
    freeze = json.loads(manifest_path.read_text())
    acceptance = json.loads(args.sampling_acceptance.read_text())
    thresholds = freeze["fixed_acceptance_thresholds"]
    if int(thresholds["bootstrap_replicates"]) != args.replicates:
        raise ValueError("replicate count differs from the frozen threshold")
    if thresholds["bootstrap_block_rule"] != "max(5, ceil(2*tau_int))":
        raise ValueError("unsupported frozen bootstrap block rule")
    if acceptance["status"] != "passed" or not acceptance["passes_sampling_gate"]:
        raise ValueError("point-estimate sampling gate did not pass")
    physical_operator_fc = None
    if args.l0_freeze_manifest:
        if freeze.get("status") != "frozen_for_L0_development":
            raise ValueError("invalid L0 freeze manifest")
        if args.temperature not in freeze.get("development_temperatures_K", []):
            raise ValueError("bootstrap temperature is outside L0 development")
        provenance = acceptance.get("l0_freeze_manifest", {})
        operator_record = freeze.get("operators", {}).get(str(args.temperature), {})
        if operator_record.get("sha256") != sha256(args.operator):
            raise ValueError("operator differs from the frozen L0 temperature lane")
        with np.load(args.operator, allow_pickle=False) as operator_data:
            physical_operator_fc = np.asarray(
                operator_data["delta_fc_full"], float
            )
        frozen_minimum = int(
            freeze["trajectory_protocol"]["final_snapshots_per_seed"]
        )
        manifest_kind = "L0_development"
    else:
        provenance = acceptance.get("freeze_manifest", {})
        frozen_minimum = int(freeze["T450_on_policy"]["snapshots_per_seed"])
        manifest_kind = "conditioned_holdout"
    if sha256(manifest_path) != provenance.get("sha256"):
        raise ValueError("sampling acceptance refers to a different freeze manifest")
    if sha256(args.operator) != acceptance["sampling_operator"]["sha256"]:
        raise ValueError("sampling operator differs from point-estimate acceptance")

    phonon = fm.load_ph(args.background)
    primitive = phonopy_to_ase(phonon.unitcell)
    primitive.wrap()
    ideal = phonopy_to_ase(phonon.supercell)
    ideal.wrap()
    cluster_space = ClusterSpace(primitive, [args.cutoff2])
    basis = fc_basis(cluster_space, ideal)

    seed_data = {}
    tau_records = []
    replay_records = []
    acceptance_checkpoints = {int(item["seed"]): item for item in acceptance["checkpoints"]}
    expected_counts = {int(item["n_snapshots"]) for item in acceptance["checkpoints"]}
    if len(expected_counts) != 1:
        raise ValueError("all three point-estimate trajectories must have equal snapshot counts")
    expected_snapshot_count = expected_counts.pop()
    if expected_snapshot_count < frozen_minimum:
        raise ValueError("sampling acceptance contains fewer than the frozen minimum snapshots")
    inverse_ideal_cell = np.linalg.inv(np.asarray(ideal.cell))
    for seed in (0, 1, 2):
        path = checkpoint_paths[seed]
        if sha256(path) != acceptance_checkpoints[seed]["snapshots_sha256"]:
            raise ValueError(f"seed {seed} checkpoint SHA-256 changed")
        snapshots = tdp._load_checkpoint_snapshots(path, ideal)
        if len(snapshots) != expected_snapshot_count:
            raise ValueError(
                f"seed {seed} contains {len(snapshots)} snapshots, "
                f"expected {expected_snapshot_count}"
            )
        temperatures = np.asarray([snapshot.get_temperature() for snapshot in snapshots], float)
        energies = np.asarray([snapshot.get_potential_energy() for snapshot in snapshots], float)
        force_rms = np.asarray(
            [np.sqrt(np.mean(np.asarray(snapshot.get_forces(), float) ** 2)) for snapshot in snapshots]
        )
        displacement_rms = []
        for snapshot in snapshots:
            displacement = np.asarray(snapshot.positions) - np.asarray(ideal.positions)
            fractional = displacement @ inverse_ideal_cell
            fractional -= np.round(fractional)
            displacement_rms.append(float(np.sqrt(np.mean((fractional @ ideal.cell) ** 2))))
        observables = (
            temperatures,
            energies / len(ideal),
            force_rms,
            np.asarray(displacement_rms),
        )
        tau_values = [integrated_autocorrelation_time(values) for values in observables]
        block_length = max(5, math.ceil(2.0 * max(tau_values)))
        block_length = min(block_length, len(snapshots))
        tau_records.append(
            {
                "seed": seed,
                "tau_int_snapshots": dict(zip(OBSERVABLE_LABELS, tau_values, strict=True)),
                "block_length_snapshots": block_length,
            }
        )

        short_snapshots, _ = subtract_operator(snapshots, ideal, args.operator)
        prepared = prepare_structures(short_snapshots, ideal)
        container = StructureContainer(cluster_space)
        for structure in prepared:
            container.add_structure(structure)
        fit_matrix, target = container.get_fit_data()
        fit_matrix = np.asarray(fit_matrix, float)
        target = np.asarray(target, float)
        if len(fit_matrix) % len(snapshots):
            raise ValueError("fit rows do not divide evenly among snapshots")
        rows_per_snapshot = len(fit_matrix) // len(snapshots)
        matrix_blocks = fit_matrix.reshape(len(snapshots), rows_per_snapshot, -1)
        target_blocks = target.reshape(len(snapshots), rows_per_snapshot)
        gram_blocks = np.einsum("nrp,nrq->npq", matrix_blocks, matrix_blocks, optimize=True)
        rhs_blocks = np.einsum("nrp,nr->np", matrix_blocks, target_blocks, optimize=True)
        full_parameters = solve_parameters(
            gram_blocks, rhs_blocks, np.arange(len(snapshots), dtype=int)
        )
        distance, fitted_frequency = spectrum(phonon, basis, full_parameters, args.npoints)
        reference_distance, reference_frequency = load_reference_spectrum(
            reference_paths[seed], args.temperature
        )
        if not np.allclose(distance, reference_distance, rtol=0.0, atol=1e-12):
            raise ValueError(f"seed {seed} band path differs from saved point estimate")
        replay_max = float(np.max(np.abs(fitted_frequency - reference_frequency)))
        if replay_max > 1.0e-4:
            raise ValueError(
                f"seed {seed} sufficient-statistics replay differs by {replay_max} cm^-1"
            )
        replay_records.append(
            {
                "seed": seed,
                "reference_tdep": str(reference_paths[seed]),
                "reference_tdep_sha256": sha256(reference_paths[seed]),
                "max_abs_spectrum_difference_cm-1": replay_max,
                "fit_rows_per_snapshot": rows_per_snapshot,
                "n_fit_parameters": fit_matrix.shape[1],
            }
        )
        seed_data[seed] = {
            "gram_blocks": gram_blocks,
            "rhs_blocks": rhs_blocks,
            "temperatures": temperatures,
            "block_length": block_length,
        }

    raw_path = args.output.with_suffix(".npz")
    pair_metrics = np.full((args.replicates, 3), np.nan)
    temperature_errors = np.full((args.replicates, 3), np.nan)
    start = 0
    if raw_path.is_file():
        with np.load(raw_path, allow_pickle=False) as saved:
            if int(saved["replicates"]) != args.replicates or int(saved["random_seed"]) != args.random_seed:
                raise ValueError("existing bootstrap checkpoint uses different settings")
            pair_metrics[:] = np.asarray(saved["pair_full_band_MAE_cm_1"], float)
            temperature_errors[:] = np.asarray(saved["mean_temperature_relative_error"], float)
        complete = np.isfinite(pair_metrics).all(axis=1) & np.isfinite(temperature_errors).all(axis=1)
        incomplete = np.flatnonzero(~complete)
        start = int(incomplete[0]) if len(incomplete) else args.replicates
        if complete[start:].any():
            raise ValueError("bootstrap checkpoint has non-contiguous completed replicates")
        print(f"resume bootstrap at replicate {start}/{args.replicates}", flush=True)

    for replicate in range(start, args.replicates):
        frequencies = []
        for seed in (0, 1, 2):
            data = seed_data[seed]
            indices = moving_block_indices(
                len(data["temperatures"]),
                data["block_length"],
                args.random_seed + replicate * 10 + seed,
            )
            parameters = solve_parameters(data["gram_blocks"], data["rhs_blocks"], indices)
            _, frequency = spectrum(
                phonon,
                basis,
                parameters,
                args.npoints,
                additive_force_constants=physical_operator_fc,
            )
            frequencies.append(frequency)
            sampled_mean = float(np.mean(data["temperatures"][indices]))
            temperature_errors[replicate, seed] = abs(sampled_mean - args.temperature) / args.temperature
        for pair_index, (left, right) in enumerate(combinations(range(3), 2)):
            pair_metrics[replicate, pair_index] = full_band_mae(
                frequencies[left], frequencies[right]
            )
        if (replicate + 1) % args.checkpoint_every == 0 or replicate + 1 == args.replicates:
            atomic_npz(
                raw_path,
                replicates=np.array(args.replicates),
                random_seed=np.array(args.random_seed),
                pair_labels=np.asarray(PAIR_LABELS),
                pair_full_band_MAE_cm_1=pair_metrics,
                mean_temperature_relative_error=temperature_errors,
            )
            print(f"bootstrap replicate {replicate + 1}/{args.replicates}", flush=True)

    pair_threshold = float(thresholds["seed_pair_full_band_MAE_cm-1"])
    temperature_threshold = float(thresholds["MD_mean_temperature_relative_error"])
    point_pairs = {
        "-".join(str(value) for value in item["seeds"]): float(item["full_band_MAE_cm-1"])
        for item in acceptance["seed_pair_spread"]
    }
    point_temperatures = {
        int(item["seed"]): float(item["mean_temperature_relative_error"])
        for item in acceptance["seed_temperature_diagnostics"]
    }
    pair_records = []
    for index, label in enumerate(PAIR_LABELS):
        upper = percentile_higher(pair_metrics[:, index], 0.95)
        pair_records.append(
            {
                "seeds": [int(value) for value in label.split("-")],
                "point_full_band_MAE_cm-1": point_pairs[label],
                "bootstrap_median_cm-1": float(np.median(pair_metrics[:, index])),
                "bootstrap_95pct_upper_cm-1": upper,
                "threshold_cm-1": pair_threshold,
                "passes_95pct_upper": upper < pair_threshold,
            }
        )
    temperature_records = []
    for seed in (0, 1, 2):
        upper = percentile_higher(temperature_errors[:, seed], 0.95)
        temperature_records.append(
            {
                "seed": seed,
                "point_mean_temperature_relative_error": point_temperatures[seed],
                "bootstrap_median_relative_error": float(np.median(temperature_errors[:, seed])),
                "bootstrap_95pct_upper_relative_error": upper,
                "threshold_relative_error": temperature_threshold,
                "passes_95pct_upper": upper <= temperature_threshold,
            }
        )
    passed = all(item["passes_95pct_upper"] for item in pair_records + temperature_records)
    payload = {
        "status": "passed" if passed else "insufficient_sampling",
        "scope": (
            f"{args.temperature} K L0 development sampling block-bootstrap"
            if args.l0_freeze_manifest
            else "450 K sampling block-bootstrap before any completed DFT force label"
        ),
        "temperature_K": args.temperature,
        "snapshots_per_seed": expected_snapshot_count,
        "replicates": args.replicates,
        "random_seed": args.random_seed,
        "confidence_rule": "one-sided empirical 95th percentile using numpy method=higher",
        "spectrum_assembly": (
            "bootstrap short-range TDEP FC2 + exact physical electronic operator FC2"
            if args.l0_freeze_manifest
            else "short-range TDEP spectrum"
        ),
        "block_bootstrap": {
            "method": "independent circular moving-block resampling within each trajectory",
            "block_rule": thresholds["bootstrap_block_rule"],
            "tau_estimator": "initial positive autocorrelation sequence; maximum over four observables",
            "trajectory_diagnostics": tau_records,
        },
        "sufficient_statistics_replay": replay_records,
        "seed_pair_spread": pair_records,
        "seed_temperature_diagnostics": temperature_records,
        "passes_sampling_bootstrap_gate": passed,
        "freeze_manifest_kind": manifest_kind,
        "freeze_manifest_sha256": sha256(manifest_path),
        "sampling_acceptance_sha256": sha256(args.sampling_acceptance),
        "operator_sha256": sha256(args.operator),
        "raw_distributions": {"path": str(raw_path), "sha256": sha256(raw_path)},
        "next_stage": (
            "release the fixed 60-structure DFT force-label shards"
            if passed
            else "keep DFT force labels locked and extend the frozen trajectories"
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
