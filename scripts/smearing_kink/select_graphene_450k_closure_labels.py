"""Freeze the 450 K closure label selection on the matched shared-initial ensemble.

R1'/R2A matched sampling for the finite-temperature DFT-TDEP reference closure.

Matched pool: ``R2AT_s0_shared_initial_sensitivity/formal_T450_Tel300``
(order-safe 450 K fixed-smearing S0 ensemble).  Its ``xats_pop1`` is
bit-exact with the paired R2AO-side run ``R2AP_fixed_smearing_SSCHA/
formal_T450``, whose stored per-config forces/energies (Ry/A, Ry) supply a
frozen pre-DFT reference for order checks.  The deployed R2AR ensemble lives
on the offline 2060 host; the shared-initial K A' frequency differs from it
by 0.068 cm-1 (accepted substitution, recorded in the manifest).

Protocol deviation from the original R1 screen (2026-08-22), documented in
the manifest: the MACE checkpoint committee is not reachable from this host
(delta checkpoint only on the offline 2060), so the
``high_committee_disagreement`` stratum is replaced by a high
R2AO-force-RMS stratum, and every S0-specific force gate is deferred; the
R2AO stored forces provide the order-safe reference instead.

No DFT label is read; the selection and the three reserved R2A batches are
frozen before any DFT is submitted.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
from audit_graphene_current_s0_fixed_smearing import (  # noqa: E402
    folded_k_aprime_mode,
    load_operator,
    periodic_assignment,
)

BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
R2R = BASE / "R2R_multipolar_background"
SHARED = R2R / "R2AT_s0_shared_initial_sensitivity/formal_T450_Tel300"
PAIRED = R2R / "R2AP_fixed_smearing_SSCHA/formal_T450"
OPERATOR = ROOT / "data/graphene_r2c_eval/operators/T300_operator.npz"
BACKGROUND = ROOT / "results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
OUT = BASE / "R1_450k_matched_overlap_screen"

RY_TO_EV = 13.605691932782346
CONDITION = {
    "lattice_temperature_K": 450.0,
    "operator_temperature_K": 300.0,
    "smearing": "fermi-dirac",
    "degauss_Ry": 0.0019000869380739254,
}
DFT_SETTINGS = {
    "kgrid": [8, 8, 1],
    "ecutwfc_Ry": 60.0,
    "ecutrho_Ry": 240.0,
    "disk_io": "none",
    "pseudopotential": "C_ONCV_PBE-1.2.upf",
    "conv_thr": 1.0e-10,
    "mixing_beta": 0.3,
}
GATES = {
    "DFT_force_component_RMSE_vs_R2AO_reference_meV_A": 30.0,
    "DFT_force_component_max_abs_vs_R2AO_reference_meV_A": 200.0,
    "Aprime_projected_DFT_restoring_slope_relative_error": 0.05,
    "Aprime_projected_R2AO_vs_DFT_regression_slope": [0.9, 1.1],
    "note_S0_specific_force_energy_ESS_gates": (
        "deferred to the offline 2060 host; the R2AO stored forces bound the "
        "S0-DFT difference through |S0-DFT| <= |S0-R2AO| + |R2AO-DFT|"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_standardize(matrix: np.ndarray) -> np.ndarray:
    median = np.median(matrix, axis=0)
    quarter = np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)
    quarter = np.where(quarter > 1.0e-12, quarter, 1.0)
    return (matrix - median) / quarter


def mic_displacement(positions: np.ndarray, reference: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Row-wise minimum-image displacement ``positions - reference``."""
    delta = positions - reference
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return fractional @ cell


def farthest_point(
    candidates: np.ndarray,
    pool: list[int],
    n_pick: int,
    seed_index: int,
) -> list[int]:
    """Greedy farthest-point order restricted to ``pool``."""
    picked = [pool[seed_index]]
    remaining = [i for i in pool if i != picked[0]]
    while remaining and len(picked) < n_pick:
        best = max(
            remaining,
            key=lambda i: min(
                np.linalg.norm(candidates[i] - candidates[p]) for p in picked
            ),
        )
        picked.append(best)
        remaining.remove(best)
    return picked


def main() -> int:
    acceptance = json.loads((SHARED / "acceptance.json").read_text())
    paired_acceptance = json.loads((PAIRED / "acceptance.json").read_text())

    xats_path = SHARED / "ensembles/xats_pop1.npy"
    shared_result = SHARED / "result.npz"
    paired_forces_path = PAIRED / "ensembles/forces_pop1.npy"
    paired_energies_path = PAIRED / "ensembles/energies_pop1.npy"
    paired_result = PAIRED / "result.npz"

    inputs = {
        "xats": {"path": str(xats_path), "sha256": sha256(xats_path)},
        "shared_result": {
            "path": str(shared_result),
            "sha256": sha256(shared_result),
            "acceptance_result_sha256": acceptance["result_sha256"],
        },
        "paired_forces": {
            "path": str(paired_forces_path),
            "sha256": sha256(paired_forces_path),
            "units": "Ry/angstrom",
        },
        "paired_energies": {
            "path": str(paired_energies_path),
            "sha256": sha256(paired_energies_path),
            "units": "Ry",
        },
        "paired_result": {
            "path": str(paired_result),
            "sha256": sha256(paired_result),
            "acceptance_result_sha256": paired_acceptance["result_sha256"],
        },
        "operator": {"path": str(OPERATOR), "sha256": sha256(OPERATOR)},
        "background": {"path": str(BACKGROUND), "sha256": sha256(BACKGROUND)},
    }
    for key in ("shared_result", "paired_result"):
        record = inputs[key]
        if record["sha256"] != record["acceptance_result_sha256"]:
            raise ValueError(f"{key} sha256 differs from its acceptance record")
    if inputs["operator"]["sha256"] != acceptance["input_provenance"]["operator"]["sha256"]:
        raise ValueError("operator sha256 differs from the shared acceptance record")
    if inputs["background"]["sha256"] != acceptance["input_provenance"]["background"]["sha256"]:
        raise ValueError("background sha256 differs from the shared acceptance record")

    xats = np.load(xats_path)
    paired_forces = np.load(paired_forces_path)
    paired_energies = np.load(paired_energies_path)
    n_configs = len(xats)
    paired_bit_exact = bool(
        np.array_equal(xats, np.load(PAIRED / "ensembles/xats_pop1.npy"))
    )
    if not paired_bit_exact:
        raise ValueError("shared-initial and R2AP xats are not bit-exact")

    _, reference_phonopy, cell, _ = load_operator(OPERATOR)

    phonon = fm.load_ph(BACKGROUND)
    supercell_positions = np.asarray(phonon.supercell.positions, float)
    supercell_cell = np.asarray(phonon.supercell.cell, float)
    if not np.allclose(supercell_cell, cell, atol=2.0e-5, rtol=0.0):
        raise ValueError("background supercell and operator cell differ")
    # The operator reference is already in phonopy supercell order; the stored
    # SSCHA arrays (xats/forces/energies) are in the interleaved CellConstructor
    # structure order of the run.  Re-verify both facts numerically: the
    # ``CellConstructor_for_phonopy`` interface array reorders CC rows into
    # phonopy order (verified by the thermal-scale displacement probe below).
    identity = periodic_assignment(reference_phonopy, supercell_positions, cell)
    if not np.array_equal(identity, np.arange(72)):
        raise ValueError("operator reference is not in phonopy supercell order")
    interface = acceptance["atom_order_interface"]
    cc_to_phonopy = np.asarray(interface["CellConstructor_for_phonopy"], int)
    phonopy_to_cc = np.asarray(interface["phonopy_for_CellConstructor"], int)
    if not np.array_equal(cc_to_phonopy[phonopy_to_cc], np.arange(72)):
        raise ValueError("acceptance interface arrays are not inverse permutations")
    with np.load(shared_result, allow_pickle=False) as data:
        if not np.array_equal(
            phonopy_to_cc, np.asarray(data["phonopy_for_CellConstructor_atom_mapping"])
        ) or not np.array_equal(
            cc_to_phonopy, np.asarray(data["CellConstructor_for_phonopy_atom_mapping"])
        ):
            raise ValueError("acceptance and result.npz CC mappings disagree")
    probe = mic_displacement(xats[0][cc_to_phonopy], reference_phonopy, cell)
    if not 0.02 < float(np.sqrt((probe**2).sum(axis=1).mean())) < 0.5:
        raise ValueError("xats rows do not displace around the mapped reference")

    # Everything downstream runs in phonopy order (the QE snapshot convention).
    xats = xats[:, cc_to_phonopy, :]
    paired_forces = paired_forces[:, cc_to_phonopy, :]

    mode_phonopy, mode_provenance = folded_k_aprime_mode(
        BACKGROUND, shared_result, reference_phonopy, cell
    )

    displacements = np.array(
        [mic_displacement(xats[i], reference_phonopy, cell) for i in range(n_configs)]
    )
    aprime_coordinate = np.einsum("ia,ia->i", displacements.reshape(n_configs, -1),
                                  np.tile(mode_phonopy.reshape(-1).real, (n_configs, 1)))

    features = {
        "RMS_displacement_A": np.sqrt((displacements**2).sum(axis=2).mean(axis=1)),
        "max_displacement_A": np.linalg.norm(displacements, axis=2).max(axis=1),
        "inplane_RMS_displacement_A": np.sqrt(
            (displacements[:, :, :2] ** 2).sum(axis=2).mean(axis=1)
        ),
        "outplane_RMS_displacement_A": np.sqrt(
            (displacements[:, :, 2] ** 2).mean(axis=1)
        ),
        "Aprime_coordinate_abs_A": np.abs(aprime_coordinate),
        "R2AO_force_RMS_eV_A": np.sqrt(
            (paired_forces**2).mean(axis=(1, 2))
        )
        * RY_TO_EV,
    }
    feature_matrix = np.column_stack([features[name] for name in features])
    standardized = robust_standardize(feature_matrix)
    geometry_columns = list(range(5))

    used: set[int] = set()

    def pick(group_pool: list[int], n_pick: int, columns) -> list[int]:
        # The ensemble is an evenodd sample: configs (2k, 2k+1) are exact
        # +/- mirror pairs, so selecting both wastes a label; the partner of
        # every pick is excluded together with the pick itself.
        sub = standardized[:, columns]
        pool = [i for i in group_pool if i not in used and (i ^ 1) not in used]
        if len(pool) < n_pick:
            raise ValueError(f"pool too small: {len(pool)} < {n_pick}")
        centroid = np.median(sub[pool], axis=0)
        seed = min(pool, key=lambda i: float(np.linalg.norm(sub[i] - centroid)))
        order = farthest_point(sub, pool, n_pick, pool.index(seed))
        used.update(order)
        for index in order:
            used.add(index ^ 1)
        return order

    rms = features["RMS_displacement_A"]
    typical_pool = list(
        np.where(
            (rms >= np.percentile(rms, 25)) & (rms <= np.percentile(rms, 75))
        )[0]
    )
    force_rms = features["R2AO_force_RMS_eV_A"]
    high_force_pool = list(np.where(force_rms >= np.percentile(force_rms, 75))[0])
    aprime = features["Aprime_coordinate_abs_A"]
    low_aprime_pool = list(np.where(aprime <= np.percentile(aprime, 10))[0])
    high_aprime_pool = list(np.where(aprime >= np.percentile(aprime, 90))[0])

    groups = {
        "typical": pick(typical_pool, 4, geometry_columns),
        "high_R2AO_force_RMS": pick(high_force_pool, 4, geometry_columns + [5]),
        "low_Aprime_amplitude": pick(low_aprime_pool, 2, geometry_columns),
        "high_Aprime_amplitude": pick(high_aprime_pool, 2, geometry_columns),
    }

    shards = {"A": [], "B": []}
    for name, indices in groups.items():
        for offset, index in enumerate(indices):
            shards["A" if offset % 2 == 0 else "B"].append(index)
    for shard in shards.values():
        if len(shard) != 6:
            raise ValueError(f"unbalanced shard: {len(shard)}")

    selected = []
    for name, indices in groups.items():
        for index in indices:
            selected.append(
                {
                    "sscha_index": int(index),
                    "selection_group": name,
                    "shard": "A"
                    if index in shards["A"]
                    else "B",
                    "features": {
                        key: float(values[index]) for key, values in features.items()
                    },
                }
            )

    remaining = [i for i in range(n_configs) if i not in used]
    reserve_order = farthest_point(
        standardized, remaining, len(remaining), 0
    )
    reserved_batches = {
        f"batch_{position + 2}": [int(i) for i in reserve_order[position * 12:(position + 1) * 12]]
        for position in range(3)
    }

    OUT.mkdir(parents=True, exist_ok=True)
    shard_arrays = {}
    for shard, indices in shards.items():
        sorted_indices = sorted(indices)
        shard_arrays[shard] = {
            "positions": xats[sorted_indices],
            "cells": np.repeat(cell[None, :, :], len(sorted_indices), axis=0),
            "numbers": np.full(72, 6, dtype=int),
            "sscha_indices": np.array(sorted_indices, dtype=int),
            "selection_groups": np.array(
                [
                    next(
                        entry["selection_group"]
                        for entry in selected
                        if entry["sscha_index"] == index
                    )
                    for index in sorted_indices
                ],
                dtype="<U32",
            ),
            "R2AO_forces_eV_A": paired_forces[sorted_indices] * RY_TO_EV,
            "R2AO_energies_eV": paired_energies[sorted_indices] * RY_TO_EV,
        }
        path = OUT / f"shard_{shard}_snapshots.npz"
        np.savez(path, **shard_arrays[shard])
        print(f"wrote {path} ({len(sorted_indices)} configs)")

    manifest = {
        "status": "frozen_before_R1_DFT",
        "scope": "order-safe 450 K fixed-smearing matched closure labels (R1' screen + R2A reserve)",
        "matched_pool": {
            "ensemble": str(SHARED),
            "paired_bit_exact_ensemble": str(PAIRED),
            "paired_xats_bit_exact": paired_bit_exact,
            "deployed_R2AR_substitution_note": (
                "deployed R2AR xats live on the offline 2060 host; shared-initial "
                "K A' differs by 0.068 cm-1 (frozen acceptance comparison)"
            ),
        },
        "selection_protocol": {
            "n_candidates": n_configs,
            "n_selected": 12,
            "groups": {name: len(v) for name, v in groups.items()},
            "diversity": "robust-standardized features with farthest-point selection",
            "evenodd_mirror_rule": (
                "configs (2k, 2k+1) are exact +/- displacement mirror pairs; "
                "the 12-point screen never contains both members of a pair"
            ),
            "reserved_R2A_batches": reserved_batches,
            "DFT_labels_read_during_selection": False,
            "deviation_from_2026_08_22_R1_protocol": (
                "MACE committee checkpoints unreachable (offline 2060); "
                "high_committee_disagreement replaced by high_R2AO_force_RMS; "
                "S0-specific force/energy/ESS gates deferred and bounded via R2AO"
            ),
        },
        "condition": CONDITION,
        "fixed_DFT_settings": DFT_SETTINGS,
        "fixed_acceptance_thresholds": GATES,
        "atom_mapping": {
            "CellConstructor_to_phonopy": cc_to_phonopy.tolist(),
            "snapshot_position_order": "phonopy",
            "stored_force_order": "CellConstructor (as saved by python-sscha)",
            "mapping_source": "acceptance atom_order_interface, re-verified numerically",
        },
        "Aprime_mode": mode_provenance,
        "selected": selected,
        "shards": {
            shard: [int(index) for index in sorted(indices)]
            for shard, indices in shards.items()
        },
        "inputs": inputs,
        "unit_conversion": {"Ry_to_eV": RY_TO_EV},
        "decision_after_DFT": (
            "evaluate the frozen R2AO-reference and A' slope gates before "
            "submitting the reserved R2A batches"
        ),
    }
    manifest_path = OUT / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for shard in shard_arrays:
        manifest["outputs"] = manifest.get("outputs", {})
        manifest["outputs"][f"shard_{shard}_snapshots_sha256"] = sha256(
            OUT / f"shard_{shard}_snapshots.npz"
        )
    manifest["outputs"]["freeze_manifest_initial_sha256"] = sha256(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(json.dumps({"shards": manifest["shards"], "reserved": reserved_batches}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
