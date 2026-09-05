#!/usr/bin/env python3
"""Fit and block-bootstrap the fixed 60-label P4 DFT-TDEP target."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from hiphive import ClusterSpace, StructureContainer
from hiphive.utilities import prepare_structures

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
import td_phonon as tdp  # noqa: E402
from bootstrap_graphene_fd_conditioned_sampling import (  # noqa: E402
    fc_basis,
    moving_block_indices,
    solve_parameters,
)
from graphene_fd_p4_common import (  # noqa: E402
    CM_PER_THZ,
    P4_LINE_QPOINTS,
    atomic_json,
    atomic_npz,
    p4_key,
    qpoint_fractional,
    sha256,
)
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402


def labelled_path(value: str) -> tuple[int, Path]:
    try:
        seed, raw_path = value.split("=", 1)
        return int(seed), Path(raw_path)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed labels must use SEED=/path") from error


def point_fc2(basis: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    return np.tensordot(parameters, basis, axes=(0, 0))


def qpoint_top_cm(phonon, force_constants: np.ndarray) -> np.ndarray:
    t_values = [value for _, value in P4_LINE_QPOINTS]
    phonon.force_constants = np.asarray(force_constants, float)
    phonon.run_qpoints(qpoint_fractional(t_values), with_dynamical_matrices=False)
    frequencies = np.sort(
        np.asarray(phonon.get_qpoints_dict()["frequencies"], float), axis=1
    )
    return frequencies[:, -1] * CM_PER_THZ


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-label", type=labelled_path, action="append", required=True)
    parser.add_argument("--merge-manifest", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--sampling-bootstrap", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--temperature", type=int, default=450)
    parser.add_argument("--cutoff2", type=float, default=6.0)
    parser.add_argument("--npoints", type=int, default=201)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=450002)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--output-tdep", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--bootstrap-output", type=Path, required=True)
    parser.add_argument("--bootstrap-summary", type=Path, required=True)
    args = parser.parse_args()

    if args.replicates <= 0 or args.checkpoint_every <= 0:
        parser.error("replicates and checkpoint-every must be positive")
    seed_paths = dict(args.seed_label)
    if set(seed_paths) != {0, 1, 2} or len(args.seed_label) != 3:
        parser.error("provide exactly one label file for seeds 0, 1, and 2")
    freeze = json.loads(args.freeze_manifest.read_text())
    merge = json.loads(args.merge_manifest.read_text())
    sampling_bootstrap = json.loads(args.sampling_bootstrap.read_text())
    thresholds = freeze["fixed_acceptance_thresholds"]
    if int(thresholds["bootstrap_replicates"]) != args.replicates:
        raise ValueError("DFT-TDEP bootstrap replicate count differs from freeze")
    if thresholds["bootstrap_block_rule"] != "max(5, ceil(2*tau_int))":
        raise ValueError("unsupported frozen bootstrap block rule")
    if merge.get("status") != "complete" or int(merge["n_structures"]) != 60:
        raise ValueError("P4 merge is incomplete")
    if merge["freeze_manifest"]["sha256"] != sha256(args.freeze_manifest):
        raise ValueError("P4 merge used a different freeze manifest")
    if merge["sampling_bootstrap"]["sha256"] != sha256(args.sampling_bootstrap):
        raise ValueError("P4 merge used a different sampling bootstrap")
    if not sampling_bootstrap.get("passes_sampling_bootstrap_gate"):
        raise ValueError("sampling bootstrap did not release DFT labels")

    phonon = fm.load_ph(args.background)
    primitive = phonopy_to_ase(phonon.unitcell)
    primitive.wrap()
    ideal = phonopy_to_ase(phonon.supercell)
    ideal.wrap()
    cluster_space = ClusterSpace(primitive, [args.cutoff2])
    basis = fc_basis(cluster_space, ideal)

    expected_indices = [
        int(value) for value in freeze["T450_on_policy"]["dft_label_indices_per_seed"]
    ]
    if len(expected_indices) < 2:
        raise ValueError("P4 requires at least two labels per seed")
    index_stride = int(np.gcd.reduce(np.diff(expected_indices)))
    if index_stride <= 0:
        raise ValueError("invalid frozen label index stride")
    sampling_diagnostics = {
        int(item["seed"]): item
        for item in sampling_bootstrap["block_bootstrap"]["trajectory_diagnostics"]
    }

    gram_by_seed = {}
    rhs_by_seed = {}
    matrix_by_seed = {}
    target_by_seed = {}
    structures_by_seed = {}
    label_block_records = []
    source_records = []
    for seed in (0, 1, 2):
        path = seed_paths[seed]
        expected_output = merge["seed_outputs"][str(seed)]
        if expected_output["sha256"] != sha256(path):
            raise ValueError(f"merged seed {seed} labels changed")
        structures = read(path, index=":")
        if len(structures) != len(expected_indices):
            raise ValueError(f"seed {seed} label count changed")
        fitted_structures = []
        identities = []
        for expected_index, structure in zip(expected_indices, structures, strict=True):
            observed = (int(structure.info["trajectory_seed"]), int(structure.info["snapshot_index"]))
            if observed != (seed, expected_index):
                raise ValueError(f"label identity changed: {observed}")
            if "REF_forces" not in structure.arrays:
                raise ValueError(f"missing DFT force in {p4_key(*observed)}")
            evaluated = structure.copy()
            properties = {"forces": np.asarray(structure.arrays["REF_forces"], float)}
            if "REF_energy" in structure.info:
                properties["energy"] = float(structure.info["REF_energy"])
            evaluated.calc = SinglePointCalculator(evaluated, **properties)
            fitted_structures.append(evaluated)
            identities.append(p4_key(*observed))

        prepared = prepare_structures(fitted_structures, ideal)
        container = StructureContainer(cluster_space)
        for structure in prepared:
            container.add_structure(structure)
        fit_matrix, target = container.get_fit_data()
        fit_matrix = np.asarray(fit_matrix, float)
        target = np.asarray(target, float)
        if len(fit_matrix) % len(structures):
            raise ValueError("fit rows do not divide evenly among P4 labels")
        rows_per_structure = len(fit_matrix) // len(structures)
        matrix_blocks = fit_matrix.reshape(len(structures), rows_per_structure, -1)
        target_blocks = target.reshape(len(structures), rows_per_structure)
        gram_by_seed[seed] = np.einsum(
            "nrp,nrq->npq", matrix_blocks, matrix_blocks, optimize=True
        )
        rhs_by_seed[seed] = np.einsum(
            "nrp,nr->np", matrix_blocks, target_blocks, optimize=True
        )
        matrix_by_seed[seed] = fit_matrix
        target_by_seed[seed] = target
        structures_by_seed[seed] = structures
        snapshot_block = int(sampling_diagnostics[seed]["block_length_snapshots"])
        label_block = min(len(structures), max(1, math.ceil(snapshot_block / index_stride)))
        label_block_records.append(
            {
                "seed": seed,
                "sampling_block_length_saved_snapshots": snapshot_block,
                "label_index_stride_saved_snapshots": index_stride,
                "label_block_length": label_block,
            }
        )
        source_records.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": sha256(path),
                "identities": identities,
                "fit_rows_per_structure": rows_per_structure,
            }
        )

    gram = np.concatenate([gram_by_seed[seed] for seed in (0, 1, 2)], axis=0)
    rhs = np.concatenate([rhs_by_seed[seed] for seed in (0, 1, 2)], axis=0)
    full_indices = np.arange(len(gram), dtype=int)
    parameters = solve_parameters(gram, rhs, full_indices)
    force_constants = point_fc2(basis, parameters)
    distance, frequency, label_positions, labels = tdp.band_from_phonopy(
        phonon, force_constants, npoints=args.npoints
    )
    point_top = qpoint_top_cm(phonon, force_constants)
    full_matrix = np.concatenate([matrix_by_seed[seed] for seed in (0, 1, 2)], axis=0)
    full_target = np.concatenate([target_by_seed[seed] for seed in (0, 1, 2)], axis=0)
    fit_residual = full_matrix @ parameters - full_target
    fit_rmse = float(np.sqrt(np.mean(fit_residual**2)))
    atomic_npz(
        args.output_tdep,
        temperatures=np.array([float(args.temperature)]),
        T450_fc2=force_constants,
        T450_dist=distance,
        T450_freq=frequency,
        T450_fit_rmse_meV_A=np.array(fit_rmse * 1000.0),
        T450_line_top_cm_1=point_top,
        line_regions=np.asarray([region for region, _ in P4_LINE_QPOINTS]),
        line_t_GK=np.asarray([value for _, value in P4_LINE_QPOINTS]),
        label_positions=label_positions,
        labels=labels,
        n_structures=np.array(60),
        n_fit_parameters=np.array(len(parameters)),
    )
    point_summary = {
        "status": "complete",
        "scope": (
            "fixed 60-label 450 K P4 DFT-TDEP evaluation target; the fitted target "
            "is not fed back into the frozen predictor"
        ),
        "temperature_K": args.temperature,
        "degauss_Ry": float(freeze["T450_on_policy"]["degauss_Ry"]),
        "n_structures": 60,
        "n_structures_per_seed": len(expected_indices),
        "fit_RMSE_meV_A": fit_rmse * 1000.0,
        "n_fit_parameters": len(parameters),
        "cutoff2_A": args.cutoff2,
        "sources": source_records,
        "freeze_manifest_sha256": sha256(args.freeze_manifest),
        "merge_manifest_sha256": sha256(args.merge_manifest),
        "background": {"path": str(args.background), "sha256": sha256(args.background)},
        "output": {"path": str(args.output_tdep), "sha256": sha256(args.output_tdep)},
    }
    atomic_json(args.output_summary, point_summary)

    raw_top = np.full((args.replicates, len(P4_LINE_QPOINTS)), np.nan)
    start = 0
    if args.bootstrap_output.is_file():
        with np.load(args.bootstrap_output, allow_pickle=False) as saved:
            if int(saved["replicates"]) != args.replicates:
                raise ValueError("existing DFT-TDEP bootstrap replicate count differs")
            if int(saved["random_seed"]) != args.random_seed:
                raise ValueError("existing DFT-TDEP bootstrap random seed differs")
            if str(saved["merge_manifest_sha256"].item()) != sha256(args.merge_manifest):
                raise ValueError("existing DFT-TDEP bootstrap used different labels")
            raw_top[:] = np.asarray(saved["line_top_cm_1"], float)
        complete = np.isfinite(raw_top).all(axis=1)
        incomplete = np.flatnonzero(~complete)
        start = int(incomplete[0]) if len(incomplete) else args.replicates
        if complete[start:].any():
            raise ValueError("DFT-TDEP bootstrap checkpoint completion is non-contiguous")
        print(f"resume DFT-TDEP bootstrap at {start}/{args.replicates}", flush=True)

    block_by_seed = {item["seed"]: item["label_block_length"] for item in label_block_records}
    n_per_seed = len(expected_indices)
    for replicate in range(start, args.replicates):
        sampled = []
        for seed in (0, 1, 2):
            local = moving_block_indices(
                n_per_seed,
                block_by_seed[seed],
                args.random_seed + replicate * 10 + seed,
            )
            sampled.append(local + seed * n_per_seed)
        sampled_indices = np.concatenate(sampled)
        replicate_parameters = solve_parameters(gram, rhs, sampled_indices)
        replicate_fc = point_fc2(basis, replicate_parameters)
        raw_top[replicate] = qpoint_top_cm(phonon, replicate_fc)
        if (replicate + 1) % args.checkpoint_every == 0 or replicate + 1 == args.replicates:
            atomic_npz(
                args.bootstrap_output,
                replicates=np.array(args.replicates),
                random_seed=np.array(args.random_seed),
                line_regions=np.asarray([region for region, _ in P4_LINE_QPOINTS]),
                line_t_GK=np.asarray([value for _, value in P4_LINE_QPOINTS]),
                line_top_cm_1=raw_top,
                label_block_lengths=np.asarray(
                    [block_by_seed[seed] for seed in (0, 1, 2)], int
                ),
                merge_manifest_sha256=np.array(sha256(args.merge_manifest)),
                point_tdep_sha256=np.array(sha256(args.output_tdep)),
            )
            print(f"DFT-TDEP bootstrap replicate {replicate + 1}/{args.replicates}", flush=True)

    result = {
        "status": "complete",
        "scope": "trajectory-stratified moving-block bootstrap of the fixed P4 DFT-TDEP target",
        "replicates": args.replicates,
        "random_seed": args.random_seed,
        "confidence_rule": "one-sided empirical 95th percentile, evaluated downstream",
        "block_bootstrap": {
            "method": "independent circular moving-block resampling within each trajectory seed",
            "frozen_snapshot_rule": thresholds["bootstrap_block_rule"],
            "label_block_translation": (
                "ceil(sampling block length in saved snapshots / frozen DFT-label index stride)"
            ),
            "trajectory_diagnostics": label_block_records,
        },
        "point_tdep": {"path": str(args.output_tdep), "sha256": sha256(args.output_tdep)},
        "raw_distributions": {
            "path": str(args.bootstrap_output),
            "sha256": sha256(args.bootstrap_output),
        },
        "merge_manifest_sha256": sha256(args.merge_manifest),
        "freeze_manifest_sha256": sha256(args.freeze_manifest),
    }
    atomic_json(args.bootstrap_summary, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
