from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/smearing_kink"
sys.path.insert(0, str(SCRIPTS))

import graphene_r2r0_formal as formal  # noqa: E402
import graphene_r2r_multipolar_background as r2r  # noqa: E402


DATA = ROOT / "data/graphene_r2o_taylor_null_core"
RUN = (
    ROOT
    / "results"
    / "graphene_physics_temperature"
    / "post_p4_feasibility"
    / "R2Q_four_step_trust_region"
    / "formal_4step_seed83_rtx"
)
SAFETY_ALIAS_ATTACKS = (
    ("trainingPerformed", True),
    ("fitPerformed", False),
    ("forceLabelsUsed", 0),
    ("heldDataAccess", True),
    ("holdoutRead", False),
    ("seed1Access", 0),
    ("trainingperformed", True),
    ("fitperformed", False),
    ("forcelabelsused", 0),
    ("helddataaccess", True),
    ("holdoutread", False),
    ("seed1access", 0),
    ("training", True),
    ("fit", False),
    ("trained", 0),
    ("fitted", True),
    ("did_fit", False),
    ("was_trained", 0),
    ("force_labels_present", True),
    ("use_force_labels", False),
    ("target_forces_used", 0),
    ("targetforcesused", True),
    ("held_dataset_opened", False),
    ("seed_1_access", 0),
    ("trainingWasPerformed", True),
)


def _structures(count: int, atoms: int, config_type: str) -> list[Atoms]:
    values = []
    for index in range(count):
        item = Atoms(
            "C" * atoms,
            positions=np.zeros((atoms, 3)),
            cell=np.diag([20.0, 20.0, 20.0]),
            pbc=True,
        )
        item.positions[:, 0] = np.arange(atoms) * 0.01 + index * 1.0e-5
        item.info["config_type"] = config_type
        values.append(item)
    return values


def _thermal_structures() -> list[Atoms]:
    return (
        _structures(20, 72, "r2o_exact_e50_seed0_train")
        + _structures(36, 72, "r2o_auxiliary_T300_train")
        + _structures(36, 72, "r2o_auxiliary_T600_train")
    )


def _design_arrays(atoms: int = 72) -> dict[str, np.ndarray]:
    return {
        "fixed_energy_eV": np.zeros(1, dtype="<f8"),
        "fixed_force_eV_A": np.zeros((atoms, 3), dtype="<f8"),
        "parameter_energy_design_eV": np.linspace(-1, 1, 65, dtype="<f8"),
        "parameter_force_design_eV_A": np.arange(atoms * 3 * 65, dtype="<f8").reshape(atoms, 3, 65) * 1.0e-5,
        "b_A4": np.full(atoms, 2.0e-6, dtype="<f8"),
        "a": np.ones(atoms, dtype="<f8"),
        "c": np.full(atoms, 0.5, dtype="<f8"),
        "combined_energy_eV": np.zeros(1, dtype="<f8"),
        "combined_force_eV_A": np.zeros((atoms, 3), dtype="<f8"),
    }


def _record(index: int = 0) -> dict:
    return {
        "structure_semantic_sha256": f"{index + 1:064x}",
        "reference_semantic_sha256": "a" * 64,
        "formal_graph_semantic_sha256": "b" * 64,
        "endpoint_state_sha256": r2r.R2Q_ENDPOINT_STATE_SHA256,
        "coefficients_sha256": r2r.MECHANICS_PROBE_COEFFICIENTS_SHA256,
        "affine_component_names_sha256": r2r.AFFINE_COMPONENT_NAMES_SHA256,
        "parameter_columns": 65,
        "query_receipt": {},
        "fixed_carrier_query_receipt": {},
        "fixed_carrier_coefficients_sha256": r2r.FIXED_CARRIER_COEFFICIENT_SHA256,
        "force_labels_used": False,
        "energy_labels_used": False,
    }


def _synthetic_zerojet(node_count: int, source_hash: str, raw_hash: str) -> dict:
    probe_nodes = sorted(set(np.linspace(0, node_count - 1, 4, dtype=int).tolist()))
    return {
            "format": "graphene_r2r_background_rank0_zero_jet_audit_v3_local_direct",
            "node_count": node_count,
            "local_sample_count": 40,
            "local_Hessian_shape_per_node": [40, 3, 40, 3],
            "multipolar_gate_value_max_abs": 0.0,
            "global_all_node_Jacobian_max_abs_A-1": 0.0,
            "global_all_node_value_and_Jacobian_exact_zero": True,
            "all_nodes_covered_exactly_once": True,
            "all_local_senders_unique": True,
            "full_local_hessian_performed": True,
            "local_all_node_value_max_abs": 0.0,
            "local_all_node_Jacobian_max_abs_A-1": 0.0,
            "local_all_node_Hessian_max_abs_A-2": 0.0,
            "local_all_node_finite": True,
            "q_signature_orbits_sha256": raw_hash,
            "production_local_source_order_sha256": source_hash,
            "nonzero_local_production_probe": [
                {
                    "node": node,
                    "base_b_A4": r2r.MULTIPOLAR_BETA_A4,
                    "scale": 1.0,
                    "b_over_beta": 1.0,
                    "target_a": -math.expm1(-1.0),
                    "local_a": -math.expm1(-1.0),
                    "production_a": -math.expm1(-1.0),
                    "value_abs_difference": 0.0,
                    "mapped_gradient_max_abs_difference_A-1": 0.0,
                    "outside_local_global_gradient_max_abs_A-1": 0.0,
                    "formal_raw_probe_format": (
                        "graphene_r2r0_zero_jet_probe_raw_v1_actual_device"
                    ),
                    "raw_local_gradient_A-1": np.zeros((40, 3)).tolist(),
                    "raw_production_global_gradient_A-1": np.zeros(
                        (node_count, 3)
                    ).tolist(),
                }
                for node in probe_nodes
            ],
            "pass": False,
    }


def _passing_mechanics_arrays_and_q() -> tuple[dict[str, np.ndarray], dict, dict]:
    arrays = {
        "reference_force_eV_A": np.zeros((72, 3), dtype="<f8"),
        "reference_Hessian_eV_A2": np.zeros((216, 216), dtype="<f8"),
        "thermal0_force_eV_A": np.zeros((72, 3), dtype="<f8"),
        "thermal0_Hessian_eV_A2": np.zeros((216, 216), dtype="<f8"),
        "finite_difference_steps_A": np.asarray(formal.FD_STEPS_A, dtype="<f8"),
        "finite_difference_energy_eV": np.zeros((6, 2), dtype="<f8"),
        "finite_difference_force_coordinate_eV_A": np.zeros((6, 2), dtype="<f8"),
    }
    snapshots = {}
    zero = {}
    for reference_name, basename in (
        ("reference_6x6", "reference_6x6.xyz"),
        ("reference_8x8", "reference_8x8.xyz"),
    ):
        reference = read(DATA / basename, index=0)
        graph = r2r.fixed_reference_neighborhood(
            reference, device="cpu", dtype=torch.float64
        )
        receiver = formal._little_endian_array(graph.receiver, "<i8")
        sender = formal._little_endian_array(graph.sender, "<i8")
        offsets = [0]
        local_sender = []
        edge_index = []
        for node in range(len(reference)):
            selected = np.flatnonzero(receiver == node).astype("<i8", copy=False)
            edge_index.extend(int(value) for value in selected)
            local_sender.extend(int(value) for value in sender[selected])
            offsets.append(len(local_sender))
        values = {
            "receiver": receiver,
            "sender": sender,
            "image_integer": formal._little_endian_array(
                graph.image_integer, "<i8"
            ),
            "local_source_order_offsets": np.asarray(offsets, dtype="<i8"),
            "local_source_order_sender": np.asarray(local_sender, dtype="<i8"),
            "local_source_order_edge_index": np.asarray(edge_index, dtype="<i8"),
            "reference_distance_A": formal._little_endian_array(
                graph.reference_distance_A, "<f8"
            ),
            "weight": formal._little_endian_array(graph.weight, "<f8"),
            "normalization": formal._little_endian_array(
                graph.normalization, "<f8"
            ),
            "center_q": formal._little_endian_array(
                1.0 / graph.normalization, "<f8"
            ),
            "neighbor_q": formal._little_endian_array(
                graph.weight / graph.normalization[graph.receiver], "<f8"
            ),
        }
        source_hash = formal._source_order_semantic_sha256(values, len(reference))
        rounded_hash = formal._array_raw_sha256(
            formal._round_q_payload(values["center_q"], values["neighbor_q"])
        )
        for device_tag in ("CPU", "CUDA"):
            name = f"{reference_name}_{device_tag}"
            snapshots[name] = {
                "reference": reference_name,
                "device": "cpu" if device_tag == "CPU" else "cuda:0",
                "atom_count": len(reference),
                "edge_count": int(receiver.size),
                "local_source_order_semantic_sha256": source_hash,
                "round12_payload_sha256": rounded_hash,
                "array_raw_sha256": {
                    key: formal._array_raw_sha256(value)
                    for key, value in values.items()
                },
                "raw_q_orbit_hash_role": "diagnostic_only_not_a_gate",
            }
            zero[name] = _synthetic_zerojet(
                len(reference), source_hash, f"raw-{name}"
            )
            for key, value in values.items():
                arrays[formal._q_array_key(reference_name, device_tag, key)] = (
                    value.copy()
                )
    return arrays, snapshots, zero


def _passing_mechanics_arrays() -> dict[str, np.ndarray]:
    return _passing_mechanics_arrays_and_q()[0]


def _passing_mechanics_receipt() -> dict:
    arrays, snapshots, zero = _passing_mechanics_arrays_and_q()
    fd = formal._analyze_formal_finite_difference(
        arrays["finite_difference_steps_A"],
        arrays["finite_difference_energy_eV"],
        arrays["finite_difference_force_coordinate_eV_A"],
        force_target_eV_A=0.0,
        force_jacobian_target_eV_A2=0.0,
    )
    fd.update(
        {
            "coordinate": [2, 1],
            "base_point_provenance": {
                "formal_R2O_graph_semantic_sha256": "graph",
                "assignment_and_MIC_semantic_sha256": "assignment",
            },
            "point_provenance": [
                {
                    "h_A": step,
                    **{
                        side: {
                            "formal_R2O_graph_semantic_sha256": "graph",
                            "assignment_and_MIC_semantic_sha256": "assignment",
                        }
                        for side in ("minus", "plus")
                    },
                }
                for step in formal.FD_STEPS_A
            ],
            "all_point_graph_and_assignment_topology_exact": True,
        }
    )
    frozen_o3 = formal.FORMAL_CONTRACT["O3_rigid_transform_probe_v4"]
    rigid_receipt = {
        "selected_graph_sha256": "d" * 64,
        "derived_graph_sha256": "d" * 64,
        "baseline_graph_frozen_hash_match": True,
        "assignment_arrays_byte_equal": {
            "reference_to_source": True,
            "source_to_reference": True,
            "image_integer_reference_order": True,
        },
        "baseline_structure_full_semantic_sha256": frozen_o3[
            "baseline_structure_full_semantic_sha256"
        ],
        "baseline_reference_full_semantic_sha256": frozen_o3[
            "baseline_reference_full_semantic_sha256"
        ],
        "native_rebuild_used_for_physics": False,
        "public_reference_input_covariance_array_exact": True,
        "public_structure_input_covariance_array_exact": True,
        "internal_covariance_atol_A": r2r.RIGID_INTERNAL_COVARIANCE_ATOL_A,
        "reordered_and_reference_covariance_within_named_tolerance": {
            "ordered_positions": True,
            "ordered_cell": True,
            "adapted_reference_positions": True,
        },
        "native_background_physical_edge_multiset_equivalent": True,
        "native_background_multiset_diagnostic": {
            "numeric_differences_are_diagnostic_only": True,
        },
        "native_rebuild_diagnostic": {
            "native_physical_edge_multiset_equivalent": True,
            "numeric_differences_are_diagnostic_only": True,
            "physics_gate_authorized": False,
        },
    }
    o3_diagnostic = {
        "authorization_role": "diagnostic_only_not_physics_gate",
        "energy_abs_difference_eV": 1.0e-6,
        "force_max_abs_difference_eV_A": 2.0e-5,
        "physics_gate_authorized": False,
    }
    return {
        "mechanics": {
            "reference": {
                "energy_abs_eV": 0.0,
                "force_max_abs_eV_A": 0.0,
                "Hessian_max_abs_eV_A2": 0.0,
                "Hessian_antisymmetry_max_abs_eV_A2": 0.0,
                "Hessian_translation_ASR_max_abs_eV_A2": 0.0,
                "Weyl_Gamma_K_drift_upper_bound_cm-1": 0.0,
            },
            "corrected_carrier_vs_frozen_R2Q": {
                name: {
                    "energy_abs_difference_eV": 0.0,
                    "force_max_abs_difference_eV_A": 0.0,
                    "Hessian_max_abs_difference_eV_A2": 0.0,
                }
                for name in ("reference_6x6", "thermal92_global_index0")
            },
            "node_parity": {
                name: {
                    "energy_sum_abs_difference_eV": 0.0,
                    "position_gradient_max_abs_difference_eV_A": 0.0,
                    "signed_l0_shape": [72, 32],
                }
                for name in ("reference_6x6", "thermal92_global_index0")
            },
            "O3": {
                name: {
                    "energy_abs_difference_eV": 0.0,
                    "force_covariance_max_abs_difference_eV_A": 0.0,
                    "query_receipt": {
                        "formal_R2O_graph_mode": "rigid_transform_probe",
                        "diagnostic_native_rebuild_selected": False,
                        "rigid_transform_receipt": rigid_receipt,
                    },
                    "native_rebuild_combined_EF_diagnostic": o3_diagnostic,
                }
                for name in ("proper", "improper")
            },
            "translation": {"energy_abs_difference_eV": 0.0, "force_max_abs_difference_eV_A": 0.0},
            "pure_permutation": {"energy_abs_difference_eV": 0.0, "force_max_abs_difference_eV_A": 0.0},
            "native_wrap_order_MIC": {"energy_abs_difference_eV": 0.0, "force_max_abs_difference_eV_A": 0.0},
            "finite_difference": fd,
            "nonreference_complete_Hessian": {"antisymmetry_max_abs_eV_A2": 0.0, "translation_ASR_max_abs_eV_A2": 0.0},
            "localized_6x6_to_8x8": {
                "reference_site_mapping": {
                    "common_count": 72,
                    "only_6x6_count": 0,
                    "only_8x8_count": 56,
                    "semantic_sha256": formal.FORMAL_CONTRACT["localized_reference_site_mapping"]["semantic_sha256"],
                },
                "total_energy_abs_difference_eV": 0.0,
                "full_system_max_force_abs_difference_eV_A": 0.0,
            },
            "locality_no_wrap": {
                "reference_6x6_shortest_translation_A": 14.76,
                "reference_8x8_shortest_translation_A": 19.68,
                "interaction_diameter_A": 12.8,
                "all_reference_neighborhoods_validated_unique_sender": True,
            },
            "q_snapshots": snapshots,
            "q_portability": {"pass": False},
            "zero_jet": zero,
            "background_6A_quintic_C2": r2r.quintic_inside_cutoff_metrics(),
            "MACE_r3p2_PolynomialCutoff_C2": r2r.quintic_inside_cutoff_metrics(),
        },
        "mechanics_pass": False,
        "force_labels_used": False,
        "energy_labels_used": False,
        "geometry_loader_isolation": dict(formal.GEOMETRY_LOADER_ISOLATION),
        "can_authorize_fit_or_training": False,
    }


def _minimal_exact_safety_receipt(receipt_kind: str) -> dict:
    receipt = {
        key: None for key in formal.RECEIPT_TOP_LEVEL_KEYS[receipt_kind]
    }
    expected_format, statuses = formal.RECEIPT_FORMAT_AND_STATUS[receipt_kind]
    receipt["format"] = expected_format
    receipt["status"] = sorted(statuses)[0]
    receipt.update(
        json.loads(json.dumps(formal.RECEIPT_SAFETY_EXPECTED[receipt_kind]))
    )
    if receipt_kind == "preflight":
        receipt["payload"] = {}
    elif receipt_kind == "shard":
        receipt["thermal_receipts"] = [{}]
        receipt["harmonic_receipts"] = []
        receipt["sentinel_receipts"] = {}
    elif receipt_kind == "mechanics":
        receipt["mechanics"] = {}
    else:
        receipt["rank_condition_precheck"] = {
            "can_authorize_fit_or_training": False
        }
    return receipt


def _core_alias_container(receipt: dict, receipt_kind: str) -> dict:
    if receipt_kind == "preflight":
        return receipt["payload"]
    if receipt_kind == "shard":
        return receipt["thermal_receipts"][0]
    if receipt_kind == "mechanics":
        return receipt["mechanics"]
    return receipt["rank_condition_precheck"]


@pytest.mark.parametrize(
    "receipt_kind", ("preflight", "shard", "mechanics", "aggregate")
)
@pytest.mark.parametrize("alias,value", SAFETY_ALIAS_ATTACKS)
def test_core_safety_alias_matrix_fails_closed(
    receipt_kind: str, alias: str, value: object
) -> None:
    receipt = _minimal_exact_safety_receipt(receipt_kind)
    _core_alias_container(receipt, receipt_kind)[alias] = value
    with pytest.raises(ValueError, match="unexpected safety-like key"):
        formal._validate_receipt_safety_fields(
            receipt, f"{receipt_kind} alias matrix", receipt_kind=receipt_kind
        )


@pytest.mark.parametrize(
    "receipt_kind", ("preflight", "shard", "mechanics", "aggregate")
)
@pytest.mark.parametrize("invalid_value", (True, 0))
def test_core_known_safety_fields_require_exact_false(
    receipt_kind: str, invalid_value: object
) -> None:
    receipt = _minimal_exact_safety_receipt(receipt_kind)
    receipt["force_labels_used"] = invalid_value
    with pytest.raises(ValueError, match="safety field changed"):
        formal._validate_receipt_safety_fields(
            receipt, f"{receipt_kind} known safety", receipt_kind=receipt_kind
        )


@pytest.mark.parametrize(
    "receipt_kind", ("preflight", "shard", "mechanics", "aggregate")
)
def test_core_safety_scanner_allows_only_named_scientific_paths(
    receipt_kind: str,
) -> None:
    receipt = _minimal_exact_safety_receipt(receipt_kind)
    _core_alias_container(receipt, receipt_kind).update(
        {"train_gate": True, "training_count": 92, "force_design_rank": 65}
    )
    formal._validate_receipt_safety_fields(
        receipt, f"{receipt_kind} benign scientific keys", receipt_kind=receipt_kind
    )


def _recursive_core_fixture(receipt_kind: str, *, preflight_kind: str = "synthetic"):
    receipt = _minimal_exact_safety_receipt(receipt_kind)
    payload = {
        "scientific": {"metric": 0.0},
        "nodes": [{"index": 0}, {"index": 1}],
    }
    if receipt_kind == "preflight":
        receipt["kind"] = preflight_kind
        receipt["payload"] = payload
    elif receipt_kind == "shard":
        receipt["thermal_receipts"] = [json.loads(json.dumps(payload)) for _ in range(2)]
        receipt["harmonic_receipts"] = []
        receipt["sentinel_receipts"] = {}
    elif receipt_kind == "mechanics":
        receipt["mechanics"] = payload
    else:
        receipt["rank_condition_precheck"] = {
            "can_authorize_fit_or_training": False,
            **payload,
        }
    return receipt


@pytest.mark.parametrize(
    "receipt_kind,preflight_kind",
    (
        ("preflight", "synthetic"),
        ("preflight", "real"),
        ("shard", "synthetic"),
        ("mechanics", "synthetic"),
        ("aggregate", "synthetic"),
    ),
)
def test_recursive_core_schema_rejects_every_extra_missing_and_list_item_key(
    receipt_kind: str,
    preflight_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _recursive_core_fixture(
        receipt_kind, preflight_kind=preflight_kind
    )
    variant = (
        f"preflight:{preflight_kind}"
        if receipt_kind == "preflight"
        else receipt_kind
    )
    monkeypatch.setitem(
        formal.RECURSIVE_RECEIPT_SCHEMA_SHA256,
        variant,
        formal.recursive_key_path_schema_sha256(receipt),
    )
    monkeypatch.setitem(
        formal.RECURSIVE_RECEIPT_SCHEMA_PATH_COUNT,
        variant,
        len(formal.recursive_key_path_schema(receipt)),
    )
    formal._validate_receipt_contract(
        receipt, "clean recursive core fixture", receipt_kind=receipt_kind
    )

    root = _core_alias_container(receipt, receipt_kind)
    if receipt_kind == "shard":
        root = receipt["thermal_receipts"][0]
    for key in (
        "foo",
        "seed01Access",
        "learningPerformed",
        "modelWeightsUpdated",
        "groundTruthLoaded",
    ):
        for value in (True, False, 0):
            tampered = json.loads(json.dumps(receipt))
            target = _core_alias_container(tampered, receipt_kind)
            if receipt_kind == "shard":
                target = tampered["thermal_receipts"][0]
            target["scientific"][key] = value
            with pytest.raises(ValueError, match=r"recursive (?:key-path )?schema"):
                formal._validate_receipt_contract(
                    tampered,
                    "extra recursive core key",
                    receipt_kind=receipt_kind,
                )

    missing = json.loads(json.dumps(receipt))
    missing_target = _core_alias_container(missing, receipt_kind)
    if receipt_kind == "shard":
        missing_target = missing["thermal_receipts"][0]
    del missing_target["scientific"]["metric"]
    with pytest.raises(ValueError, match=r"recursive (?:key-path )?schema"):
        formal._validate_receipt_contract(
            missing, "missing recursive core key", receipt_kind=receipt_kind
        )

    list_extra = json.loads(json.dumps(receipt))
    list_target = _core_alias_container(list_extra, receipt_kind)
    if receipt_kind == "shard":
        list_target = list_extra["thermal_receipts"][0]
    list_target["nodes"][0]["foo"] = 0
    with pytest.raises(ValueError, match="list elements have inconsistent"):
        formal._validate_receipt_contract(
            list_extra,
            "recursive core list item key",
            receipt_kind=receipt_kind,
        )


def test_recursive_schema_hash_constants_are_frozen() -> None:
    assert formal.RECURSIVE_RECEIPT_SCHEMA_SHA256 == {
        "preflight:synthetic": "ea0aa220598dc4c6517c70fcf9d75fd53aa3fe31a172719db1bc9cad8375e394",
        "preflight:real": "6f19a3efdb9793a54cc56ef42b0f0a9db1281261d0850b8c1d7a27d07c69cc07",
        "shard": "673acd629b9d7f6df15ab4a26a290674b3eebfcb0c03b3fbc900677a6db2f23e",
        "mechanics": "a87b3d99c9c83c5436b425431efceba51bdf021a3e7e0daababd596e806f39fa",
        "aggregate": "c64f92d287f98564865512943ff2341e703e0388a5ddf65f7a909ac4f8b75aaa",
    }
    assert formal.RECURSIVE_RECEIPT_SCHEMA_PATH_COUNT == {
        "preflight:synthetic": 77,
        "preflight:real": 266,
        "shard": 654,
        "mechanics": 1815,
        "aggregate": 455,
    }
    import launch_graphene_r2r0_formal as launcher

    assert launcher.LAUNCH_RECURSIVE_SCHEMA_PATH_COUNT == 211
    assert launcher.LAUNCH_RECURSIVE_SCHEMA_SHA256 == (
        "a6eb1777c1504987eba4f590f10f428cf2741ce08983587c7f25e40599633145"
    )


def test_recursive_launcher_schema_rejects_every_nested_schema_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import launch_graphene_r2r0_formal as launcher

    receipt = {
        key: None for key in launcher.LAUNCH_RECEIPT_TOP_LEVEL_KEYS
    }
    receipt.update(
        {
            "format": launcher.LAUNCH_RECEIPT_FORMAT,
            "status": formal.STATUS_AGGREGATE_INCONCLUSIVE,
            "aggregate_receipt": {
                key: None for key in launcher.LAUNCH_AGGREGATE_SUMMARY_KEYS
            },
            "fit_or_training": False,
            "held_or_support_access": False,
            "plan": {"scientific": {"metric": 0.0}},
            "commands": [
                {
                    "scientific": {"metric": 0.0},
                    "nodes": [{"index": 0}, {"index": 1}],
                }
                for _ in range(2)
            ],
        }
    )
    launcher_paths = formal.recursive_key_path_schema(
        receipt,
        wildcard_mapping_paths=launcher.LAUNCH_RECURSIVE_WILDCARD_MAPPING_PATHS,
    )
    monkeypatch.setattr(
        launcher,
        "LAUNCH_RECURSIVE_SCHEMA_SHA256",
        formal.recursive_key_path_schema_sha256(
            receipt,
            wildcard_mapping_paths=launcher.LAUNCH_RECURSIVE_WILDCARD_MAPPING_PATHS,
        ),
    )
    monkeypatch.setattr(
        launcher, "LAUNCH_RECURSIVE_SCHEMA_PATH_COUNT", len(launcher_paths)
    )
    launcher._validate_launch_safety_fields(receipt)
    for key in (
        "foo",
        "seed01Access",
        "learningPerformed",
        "modelWeightsUpdated",
        "groundTruthLoaded",
    ):
        for value in (True, False, 0):
            for path in ("plan", "commands"):
                tampered = json.loads(json.dumps(receipt))
                target = (
                    tampered["plan"]["scientific"]
                    if path == "plan"
                    else tampered["commands"][0]["scientific"]
                )
                target[key] = value
                with pytest.raises(
                    ValueError, match=r"recursive (?:key-path )?schema"
                ):
                    launcher._validate_launch_safety_fields(tampered)

    missing = json.loads(json.dumps(receipt))
    del missing["plan"]["scientific"]["metric"]
    with pytest.raises(ValueError, match=r"recursive (?:key-path )?schema"):
        launcher._validate_launch_safety_fields(missing)

    list_extra = json.loads(json.dumps(receipt))
    list_extra["commands"][0]["nodes"][0]["foo"] = 0
    with pytest.raises(ValueError, match="list elements have inconsistent"):
        launcher._validate_launch_safety_fields(list_extra)


def test_combined_ef_adapter_returns_tuple_and_forwards_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {"adapter_probe": "exact-receipt-object"}
    observed = {}

    def fake_combined(model, structure, reference, *, device, **kwargs):
        observed.update(
            model=model,
            structure=structure,
            reference=reference,
            device=device,
            kwargs=kwargs,
        )
        return SimpleNamespace(
            energy_eV=torch.tensor(1.25, dtype=torch.float64),
            force_source_order_eV_A=torch.tensor(
                [[1.0, -2.0, 3.0]], dtype=torch.float32
            ),
            query_receipt=receipt,
        )

    monkeypatch.setattr(r2r, "production_combined_energy_force", fake_combined)
    model = object()
    structure = Atoms("C", positions=[[0.0, 0.0, 0.0]])
    reference = structure.copy()
    result = formal._combined_ef(
        model, structure, reference, "cpu", graph_mode="baseline"
    )
    assert isinstance(result, tuple) and len(result) == 3
    energy, force, returned_receipt = result
    assert energy == 1.25
    assert force.dtype == np.float64
    assert np.array_equal(force, np.asarray([[1.0, -2.0, 3.0]]))
    assert returned_receipt is receipt
    assert observed == {
        "model": model,
        "structure": structure,
        "reference": reference,
        "device": "cpu",
        "kwargs": {"graph_mode": "baseline"},
    }


def test_o3_covariant_provenance_helper_is_independent_and_fail_closed() -> None:
    item = _passing_mechanics_receipt()["mechanics"]["O3"]["proper"]
    assert formal._o3_covariant_provenance_ok(item) is True
    tampered = json.loads(json.dumps(item))
    tampered["query_receipt"]["rigid_transform_receipt"][
        "selected_graph_sha256"
    ] = "e" * 64
    assert formal._o3_covariant_provenance_ok(tampered) is False


def test_real_combined_ef_adapter_thermal0_proper_o3() -> None:
    checkpoint = formal.r2o.torch_load(RUN / "endpoint.pt", map_location="cpu")
    model = checkpoint["model"].to(dtype=torch.float64).eval()
    reference = read(DATA / "reference_6x6.xyz")
    thermal = formal.read_geometry_only_extxyz(DATA / "train_thermal.xyz")[0]
    base_energy, base_force, base_receipt = formal._combined_ef(
        model, thermal, reference, "cpu"
    )
    assert isinstance(base_energy, float)
    assert base_force.shape == (72, 3) and base_force.dtype == np.float64
    assert base_receipt["formal_R2O_graph_mode"] == "baseline"

    generator = np.random.default_rng(83)
    transformation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(transformation) < 0:
        transformation[:, 0] *= -1
    transformed_reference = reference.copy()
    transformed_reference.positions = np.asarray(reference.positions) @ transformation.T
    transformed_reference.set_cell(
        np.asarray(reference.cell) @ transformation.T, scale_atoms=False
    )
    transformed_thermal = thermal.copy()
    transformed_thermal.positions = np.asarray(thermal.positions) @ transformation.T
    transformed_thermal.set_cell(
        np.asarray(thermal.cell) @ transformation.T, scale_atoms=False
    )
    energy, force, query_receipt = formal._combined_ef(
        model,
        transformed_thermal,
        transformed_reference,
        "cpu",
        graph_mode="rigid_transform_probe",
        baseline_reference_template=reference,
        baseline_structure_template=thermal,
        rigid_transform=transformation,
    )
    gates = r2r.CANONICAL_CONTRACT["fixed_gates"]["O3_proper_and_improper"]
    assert abs(energy - base_energy) <= gates["energy_abs_eV"]
    assert np.max(np.abs(force - base_force @ transformation.T)) <= gates[
        "force_covariance_max_abs_eV_A"
    ]
    diagnostic = r2r.production_rigid_transform_native_rebuild_diagnostic(
        model,
        transformed_thermal,
        transformed_reference,
        baseline_reference_template=reference,
        baseline_structure_template=thermal,
        rigid_transform=transformation,
        device="cpu",
    )
    assert formal._o3_covariant_provenance_ok(
        {
            "query_receipt": query_receipt,
            "native_rebuild_combined_EF_diagnostic": diagnostic,
        }
    )


def test_frozen_modulo_shards_and_geometry_only_loader() -> None:
    o3 = formal.FORMAL_CONTRACT["O3_rigid_transform_probe_v4"]
    assert o3["baseline_structure_template"] == "thermal92 global index 0"
    assert o3["baseline_structure_full_semantic_sha256"] == (
        "d3f6eca52753a6407c58d107374e06e80cec470c3668c1c6f1f034ea59488766"
    )
    assert o3["baseline_reference_full_semantic_sha256"] == (
        "39cf74c68a5ce629ab708ce9158f49d72a7e955895d5b01fa10c2883caa861e7"
    )
    assert o3["native_rebuild"] == "diagnostic only, never the O3 physics gate"
    assert "array-exact" in o3["public_input_covariance"]
    assert "1e-12 A" in o3["public_input_covariance"]
    assert "identity-key sets only" in o3["native_multiset_veto"]
    assert [len(formal.SHARD_INDICES[i]["thermal"]) for i in range(3)] == [31, 31, 30]
    assert [len(formal.SHARD_INDICES[i]["harmonic"]) for i in range(3)] == [11, 11, 10]
    assert sorted(sum((list(formal.SHARD_INDICES[i]["thermal"]) for i in range(3)), [])) == list(range(92))
    thermal = formal.read_geometry_only_extxyz(DATA / "train_thermal.xyz")
    harmonic = formal.read_geometry_only_extxyz(
        DATA / "train_harmonic_lambda1_small_zero.xyz"
    )
    assert len(thermal) == 92 and len(harmonic) == 32
    assert set(thermal[0].arrays) == {"numbers", "positions"}
    assert set(thermal[0].info) == {"config_type"}
    assert set(harmonic[0].arrays) == {"numbers", "positions"}


def test_geometry_loader_discards_malicious_label_text_without_conversion(tmp_path: Path) -> None:
    source = tmp_path / "malicious_train.xyz"
    source.write_text(
        "1\n"
        'Lattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3:forces:R:3 '
        'pbc="T T T" config_type=safe energy=NOT_A_NUMBER virial="DO NOT RETAIN" stress=BAD\n'
        "C 0 0 0 force_x force_y force_z\n",
        encoding="utf-8",
    )
    structures = formal.read_geometry_only_extxyz(source)
    assert len(structures) == 1
    assert set(structures[0].arrays) == {"numbers", "positions"}
    assert structures[0].info == {"config_type": "safe"}


def test_path_policy_allows_legal_small_zero_and_rejects_held_tokens(tmp_path: Path) -> None:
    legal = tmp_path / "train_harmonic_lambda1_small_zero.xyz"
    legal.touch()
    assert formal._reject_path(legal, "legal") == legal.resolve()
    with pytest.raises(ValueError, match="forbidden R2R-0 path token"):
        formal._reject_path(tmp_path / "seed1" / "x.xyz", "illegal", must_exist=False)
    with pytest.raises(ValueError, match="forbidden R2R-0 path token"):
        formal._reject_path(tmp_path / "small12" / "x.xyz", "illegal", must_exist=False)
    with pytest.raises(ValueError, match="traversal"):
        formal._reject_path(tmp_path / "plain" / ".." / "x.xyz", "illegal", must_exist=False)
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        formal._reject_path(link / "x.xyz", "illegal", must_exist=False)


def test_localized_reference_mapping_is_identity_bound() -> None:
    mapping = formal.localized_reference_site_mapping(
        read(DATA / "reference_6x6.xyz"), read(DATA / "reference_8x8.xyz")
    )
    assert mapping["common_count"] == 72
    assert mapping["only_6x6_count"] == 0
    assert mapping["only_8x8_count"] == 56
    assert mapping["semantic_sha256"] == (
        "5f12c337580f80428235644b196d703f7aa8affb426beec3f962d7e6861230c1"
    )


def test_sentinel_symmetric_tolerance_and_four_probes() -> None:
    shards = []
    for shard in range(3):
        arrays = {}
        receipts = {}
        for index in formal.SENTINEL_GLOBAL_INDICES:
            receipt = _record(index)
            receipt["structure_semantic_sha256"] = f"structure-{index}"
            receipts[str(index)] = {"global_index": index, **receipt}
            for key, value in _design_arrays().items():
                arrays[f"sentinel_{index}_{key}"] = value.copy()
        shards.append(({"sentinel_receipts": receipts}, arrays))
    result = formal.compare_sentinels(shards)
    assert result["pass"] is True
    assert len(result["comparisons"]) == 12
    shards[2][1]["sentinel_56_parameter_force_design_eV_A"][0, 0, 0] += 1.0e-4
    assert formal.compare_sentinels(shards)["pass"] is False


def test_shard_array_contract_is_exact_and_finite() -> None:
    design = np.zeros((92, 72, 3, 65), dtype="<f8")
    receipt, arrays = _synthetic_loaded_shard(0, design, {})
    formal._validate_shard_array_contract(receipt, arrays)
    broken = dict(arrays)
    broken.pop("thermal_c")
    with pytest.raises(ValueError, match="key set"):
        formal._validate_shard_array_contract(receipt, broken)
    nonfinite = dict(arrays)
    nonfinite["thermal_fixed_energy_eV"] = arrays["thermal_fixed_energy_eV"].copy()
    nonfinite["thermal_fixed_energy_eV"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        formal._validate_shard_array_contract(receipt, nonfinite)


def test_mechanics_gate_is_recomputed_not_trusted_boolean() -> None:
    receipt = _passing_mechanics_receipt()
    arrays = _passing_mechanics_arrays()
    assert receipt["mechanics_pass"] is False
    assert formal.recompute_mechanics_gate(receipt, arrays)["pass"] is True
    receipt["mechanics"]["translation"]["energy_abs_difference_eV"] = 1.0
    recomputed = formal.recompute_mechanics_gate(receipt, arrays)
    assert recomputed["pass"] is False
    assert recomputed["checks"]["translation"] is False


def test_attempt3_fd_uses_fixed_5e5_and_retains_six_step_report() -> None:
    receipt = _passing_mechanics_receipt()
    arrays = _passing_mechanics_arrays()
    assert formal.FD_STEPS_A == (
        8.0e-4,
        4.0e-4,
        2.0e-4,
        1.0e-4,
        5.0e-5,
        2.5e-5,
    )
    thresholds = r2r.CANONICAL_CONTRACT["fixed_gates"]["finite_difference"]
    assert thresholds == {
        "force_abs_eV_A": 1.0e-5,
        "force_Hessian_abs_eV_A2": 1.0e-5,
    }
    # A deliberately bad report-only h=1e-4 point cannot select itself.
    report_index = list(formal.FD_STEPS_A).index(1.0e-4)
    report_h = formal.FD_STEPS_A[report_index]
    arrays["finite_difference_energy_eV"][report_index, 1] = -2 * report_h * 1.0e-3
    arrays["finite_difference_force_coordinate_eV_A"][report_index, 1] = (
        2 * report_h * 1.0e-3
    )
    recomputed = formal.recompute_mechanics_gate(receipt, arrays)
    assert recomputed["checks"]["finite_difference"] is True
    assert len(recomputed["finite_difference_recomputed"]["points"]) == 6
    assert (
        recomputed["finite_difference_recomputed"]["adjudicating_step_A"]
        == 5.0e-5
    )
    selected_index = list(formal.FD_STEPS_A).index(5.0e-5)
    selected_h = formal.FD_STEPS_A[selected_index]
    arrays["finite_difference_energy_eV"][selected_index, 1] = (
        -2 * selected_h * 2.0e-5
    )
    assert formal.recompute_mechanics_gate(receipt, arrays)["checks"][
        "finite_difference"
    ] is False


def test_q_portability_allows_raw_hash_difference_and_recomputes_arrays() -> None:
    receipt = _passing_mechanics_receipt()
    arrays = _passing_mechanics_arrays()
    recomputed = formal.recompute_mechanics_gate(receipt, arrays)
    q = recomputed["q_portability_recomputed"]
    assert q["pass"] is True
    assert all(
        item["raw_q_orbit_hash_equal"] is False
        and item["raw_q_orbit_hash_role"] == "diagnostic_only_not_a_gate"
        for item in q["per_reference"].values()
    )

    small = {key: value.copy() for key, value in arrays.items()}
    key = formal._q_array_key("reference_6x6", "CUDA", "center_q")
    small[key][0] = np.nextafter(small[key][0], np.inf)
    # Rebind the snapshot to prove a nonzero raw difference below both gates is legal.
    snapshot = receipt["mechanics"]["q_snapshots"]["reference_6x6_CUDA"]
    snapshot["array_raw_sha256"]["center_q"] = formal._array_raw_sha256(small[key])
    snapshot["round12_payload_sha256"] = formal._array_raw_sha256(
        formal._round_q_payload(
            small[key],
            small[formal._q_array_key("reference_6x6", "CUDA", "neighbor_q")],
        )
    )
    q_small = formal.recompute_mechanics_gate(receipt, small)[
        "q_portability_recomputed"
    ]["per_reference"]["reference_6x6"]
    assert 0.0 < q_small["normalized_q_max_abs"] <= formal.Q_MAX_ABS_TOLERANCE
    assert q_small["pass"] is True


def test_q_portability_fails_each_required_binding() -> None:
    def passing() -> tuple[dict, dict[str, np.ndarray]]:
        return _passing_mechanics_receipt(), _passing_mechanics_arrays()

    receipt, arrays = passing()
    sender = formal._q_array_key("reference_6x6", "CUDA", "sender")
    arrays[sender][0] += 1
    assert formal.recompute_mechanics_gate(receipt, arrays)["checks"]["zero_jet"] is False

    receipt, arrays = passing()
    center = formal._q_array_key("reference_6x6", "CUDA", "center_q")
    arrays[center][0] += 1.0e-11
    assert formal.recompute_mechanics_gate(receipt, arrays)["checks"]["zero_jet"] is False

    for field, value in (
        ("production_local_source_order_sha256", "changed"),
        ("node_count", 71),
        ("local_sample_count", 39),
        ("local_Hessian_shape_per_node", [39, 3, 39, 3]),
    ):
        receipt, arrays = passing()
        receipt["mechanics"]["zero_jet"]["reference_6x6_CPU"][field] = value
        assert formal.recompute_mechanics_gate(receipt, arrays)["checks"][
            "zero_jet"
        ] is False

    receipt, arrays = passing()
    receipt["mechanics"]["q_snapshots"]["reference_6x6_CPU"][
        "array_raw_sha256"
    ]["center_q"] = "changed"
    assert formal.recompute_mechanics_gate(receipt, arrays)["checks"]["zero_jet"] is False

    receipt, arrays = passing()
    value_probe = receipt["mechanics"]["zero_jet"]["reference_6x6_CPU"][
        "nonzero_local_production_probe"
    ][0]
    value_probe["production_a"] = 123.0
    value_probe["value_abs_difference"] = 0.0
    assert formal.recompute_mechanics_gate(receipt, arrays)["checks"]["zero_jet"] is False

    receipt, arrays = passing()
    gradient_probe = receipt["mechanics"]["zero_jet"]["reference_6x6_CPU"][
        "nonzero_local_production_probe"
    ][0]
    gradient_probe["raw_production_global_gradient_A-1"][0][0] = 1.0
    gradient_probe["mapped_gradient_max_abs_difference_A-1"] = 0.0
    assert formal.recompute_mechanics_gate(receipt, arrays)["checks"]["zero_jet"] is False


def test_formal_q_capture_divides_on_graph_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    sender = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    graph = SimpleNamespace(
        receiver=receiver,
        sender=sender,
        image_integer=torch.zeros((4, 3), dtype=torch.long),
        reference_distance_A=torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float64),
        weight=torch.tensor([0.5, 0.25, 0.5, 0.25], dtype=torch.float64),
        normalization=torch.tensor([1.75, 1.75], dtype=torch.float64),
    )
    source_payload = [
        {"receiver": 0, "sender_order": [0, 1]},
        {"receiver": 1, "sender_order": [0, 1]},
    ]
    monkeypatch.setattr(
        formal.r2r, "fixed_reference_neighborhood", lambda *args, **kwargs: graph
    )
    monkeypatch.setattr(
        formal.r2r,
        "background_rank0_zero_jet_audit",
        lambda *args, **kwargs: {
            "production_local_source_order_sha256": r2r.semantic_sha256(
                source_payload
            )
        },
    )
    monkeypatch.setattr(formal, "_attach_formal_zero_jet_probe_raw", lambda *_: None)
    public, arrays = formal._capture_formal_q_snapshot(
        Atoms("CC"), "reference_2", "cpu"
    )
    np.testing.assert_array_equal(
        arrays["center_q"], (1.0 / graph.normalization).numpy()
    )
    np.testing.assert_array_equal(
        arrays["neighbor_q"],
        (graph.weight / graph.normalization[graph.receiver]).numpy(),
    )
    assert public["local_source_order_semantic_sha256"] == r2r.semantic_sha256(
        source_payload
    )


def test_attempt3_mechanics_array_schema_is_exact() -> None:
    arrays = _passing_mechanics_arrays()
    expected = formal._mechanics_expected_array_schema()
    assert len(expected) == 51
    assert set(arrays) == set(expected)
    for key, (shape, dtype) in expected.items():
        assert arrays[key].shape == shape
        assert arrays[key].dtype == np.dtype(dtype)


def test_external_freeze_manifest_and_marker_are_exact(tmp_path: Path) -> None:
    manifest = tmp_path / "freeze_manifest.json"
    payload = formal.prepare_freeze_manifest(manifest)
    assert payload["authorization_marker_created"] is False
    assert not (tmp_path / "R2R0_FORMAL_GO").exists()
    marker = tmp_path / "R2R0_FORMAL_GO"
    marker.write_text(formal.sha256(manifest) + "\n", encoding="ascii")
    authorization = formal.validate_execution_authorization(manifest, marker)
    assert authorization["freeze_manifest_sha256"] == formal.sha256(manifest)
    marker.write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="does not bind"):
        formal.validate_execution_authorization(manifest, marker)


def _synthetic_loaded_shard(shard_id: int, full_design: np.ndarray, authorization: dict):
    thermal_indices = np.asarray(formal.SHARD_INDICES[shard_id]["thermal"], dtype="<i8")
    harmonic_indices = np.asarray(formal.SHARD_INDICES[shard_id]["harmonic"], dtype="<i8")
    thermal_records = [
        {"global_index": int(index), **_record(int(index))} for index in thermal_indices
    ]
    harmonic_records = [
        {"global_index": int(index), **_record(int(index))} for index in harmonic_indices
    ]
    arrays = {
        "thermal_global_index": thermal_indices,
        "thermal_fixed_energy_eV": np.zeros((len(thermal_indices), 1), dtype="<f8"),
        "thermal_fixed_force_eV_A": np.zeros((len(thermal_indices), 72, 3), dtype="<f8"),
        "thermal_parameter_energy_design_eV": np.zeros((len(thermal_indices), 65), dtype="<f8"),
        "thermal_parameter_force_design_eV_A": full_design[thermal_indices].astype("<f8"),
        "thermal_b_A4": np.full((len(thermal_indices), 72), 2.0e-6, dtype="<f8"),
        "thermal_a": np.ones((len(thermal_indices), 72), dtype="<f8"),
        "thermal_c": np.full((len(thermal_indices), 72), 0.5, dtype="<f8"),
        "thermal_combined_energy_eV": np.zeros((len(thermal_indices), 1), dtype="<f8"),
        "thermal_combined_force_eV_A": np.zeros((len(thermal_indices), 72, 3), dtype="<f8"),
        "harmonic_global_index": harmonic_indices,
        "harmonic_b_A4": np.zeros((len(harmonic_indices), 128), dtype="<f8"),
        "harmonic_a": np.zeros((len(harmonic_indices), 128), dtype="<f8"),
        "harmonic_c": np.zeros((len(harmonic_indices), 128), dtype="<f8"),
        "harmonic_fixed_carrier_energy_eV": np.zeros((len(harmonic_indices), 1), dtype="<f8"),
        "harmonic_fixed_carrier_force_eV_A": np.zeros((len(harmonic_indices), 128, 3), dtype="<f8"),
        "harmonic_combined_energy_eV": np.zeros((len(harmonic_indices), 1), dtype="<f8"),
        "harmonic_combined_force_eV_A": np.zeros((len(harmonic_indices), 128, 3), dtype="<f8"),
    }
    sentinel_receipts = {}
    for index in formal.SENTINEL_GLOBAL_INDICES:
        sentinel_receipts[str(index)] = {"global_index": index, **_record(index)}
        for key, value in _design_arrays().items():
            arrays[f"sentinel_{index}_{key}"] = value.copy()
    receipt = {
        "format": formal.SHARD_FORMAT,
        "status": formal.STATUS_SHARD,
        "formal_contract_sha256": formal.FORMAL_CONTRACT_SHA256,
        "execution_authorization": authorization,
        "frozen_source_sha256": {"source": "hash"},
        "frozen_R2R_canonical_sha256": formal.FROZEN_R2R_CANONICAL_SHA256,
        "input_sha256": {"input": "hash"},
        "runtime_fingerprint": {"semantic_sha256": f"env-{shard_id}"},
        "shard_id": shard_id,
        "partition": formal.SHARD_INDICES[shard_id],
        "thermal_count": len(thermal_indices),
        "harmonic_count": len(harmonic_indices),
        "thermal_receipts": thermal_records,
        "harmonic_receipts": harmonic_records,
        "sentinel_receipts": sentinel_receipts,
        "sentinel_included_in_matrix": False,
        "sentinel_contract_sha256": formal.SENTINEL_CONTRACT_SHA256,
        "array_artifacts": {},
        "elapsed_seconds": 1.0,
        "force_labels_used": False,
        "energy_labels_used": False,
        "geometry_loader_isolation": dict(formal.GEOMETRY_LOADER_ISOLATION),
        "can_authorize_fit_or_training": False,
    }
    return receipt, arrays


def test_synthetic_aggregate_recomputes_all_inputs_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This scientific fixture deliberately uses compact mock query receipts.
    # Recursive production schemas are exercised independently below.
    monkeypatch.setattr(
        formal, "_validate_recursive_receipt_schema", lambda *_args, **_kwargs: None
    )
    rng = np.random.default_rng(83065)
    full_design = rng.normal(size=(92, 72, 3, 65)).astype("<f8")
    authorization = {
        "freeze_manifest_sha256": "m",
        "authorization_marker_sha256": "g",
        "formal_source_sha256": {
            name: formal.sha256(path) for name, path in formal.FORMAL_SOURCE_PATHS.items()
        },
    }
    loaded = [_synthetic_loaded_shard(i, full_design, authorization) for i in range(3)]
    shard_roots = []
    for index in range(3):
        root = tmp_path / f"shard{index}"
        root.mkdir()
        (root / "receipt.json").write_text(f"{index}\n", encoding="ascii")
        shard_roots.append(root)
    mechanics_root = tmp_path / "mechanics"
    mechanics_root.mkdir()
    (mechanics_root / "receipt.json").write_text("mechanics\n", encoding="ascii")
    mechanics_arrays = _passing_mechanics_arrays()
    mechanics = {
        **_passing_mechanics_receipt(),
        "format": formal.MECHANICS_FORMAT,
        "status": formal.STATUS_MECHANICS,
        "formal_contract_sha256": formal.FORMAL_CONTRACT_SHA256,
        "execution_authorization": authorization,
        "frozen_source_sha256": {"source": "hash"},
        "input_sha256": {"input": "hash"},
        "runtime_fingerprint": {"semantic_sha256": "mechanics-env"},
        "full_H_stage_receipts": [],
        "cuda_peak_allocated_bytes": 0,
        "dtype": "torch.float64",
        "arrays_sha256": "synthetic",
        "array_schema": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in mechanics_arrays.items()
        },
        "elapsed_seconds": 1.0,
    }
    monkeypatch.setattr(formal, "validate_execution_authorization", lambda *_args: authorization)
    monkeypatch.setattr(formal, "validate_frozen_sources", lambda: {"source": "hash"})
    monkeypatch.setattr(formal, "expected_input_sha256", lambda: {"input": "hash"})
    monkeypatch.setattr(
        formal,
        "_load_completed_shard",
        lambda path: loaded[int(Path(path).name[-1])],
    )
    monkeypatch.setattr(
        formal,
        "_load_completed_mechanics",
        lambda _path: (mechanics, mechanics_arrays),
    )
    output = tmp_path / "aggregate"
    receipt = formal.aggregate_shards(
        shard_roots,
        mechanics_root,
        output,
        freeze_manifest=tmp_path / "manifest.json",
        authorization_marker=tmp_path / "GO",
    )
    assert receipt["status"] == formal.STATUS_AGGREGATE_PASS
    assert receipt["mechanics_recomputed_gate"]["pass"] is True
    assert receipt["rank_condition_precheck"]["scaled_rank"] == 65
    recovered = formal.aggregate_shards(
        shard_roots,
        mechanics_root,
        output,
        freeze_manifest=tmp_path / "manifest.json",
        authorization_marker=tmp_path / "GO",
    )
    assert recovered["completion_recovered_without_recompute"] is True
    receipt_path = output / "receipt.json"
    arrays_path = output / "arrays.npz"
    original_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    original_arrays_bytes = arrays_path.read_bytes()

    def write_recovery_receipt(candidate: dict) -> None:
        formal._atomic_json(receipt_path, candidate)
        (output / "DONE").write_text(
            candidate["status"] + "\n" + formal.sha256(receipt_path) + "\n",
            encoding="ascii",
        )

    for safety_field in (
        "can_authorize_fit_or_training",
        "fit_performed",
        "force_labels_used",
        "energy_labels_used",
        "held_or_support_access",
        "seed1_or_seed2_access",
    ):
        safety_tamper = json.loads(json.dumps(original_receipt))
        safety_tamper[safety_field] = True
        write_recovery_receipt(safety_tamper)
        with pytest.raises(ValueError, match="safety field changed"):
            formal.aggregate_shards(
                shard_roots,
                mechanics_root,
                output,
                freeze_manifest=tmp_path / "manifest.json",
                authorization_marker=tmp_path / "GO",
            )
    for geometry_field in formal.GEOMETRY_LOADER_ISOLATION:
        geometry_tamper = json.loads(json.dumps(original_receipt))
        geometry_tamper["geometry_loader_isolation"][geometry_field] = True
        write_recovery_receipt(geometry_tamper)
        with pytest.raises(ValueError, match="geometry_loader_isolation"):
            formal.aggregate_shards(
                shard_roots,
                mechanics_root,
                output,
                freeze_manifest=tmp_path / "manifest.json",
                authorization_marker=tmp_path / "GO",
            )
    for alias in ("fit_or_training", "training_performed"):
        extra_top_level = json.loads(json.dumps(original_receipt))
        extra_top_level[alias] = True
        write_recovery_receipt(extra_top_level)
        with pytest.raises(ValueError, match="top-level receipt schema changed"):
            formal.aggregate_shards(
                shard_roots,
                mechanics_root,
                output,
                freeze_manifest=tmp_path / "manifest.json",
                authorization_marker=tmp_path / "GO",
            )
    nested_alias = json.loads(json.dumps(original_receipt))
    nested_alias["rank_condition_precheck"]["training_performed"] = True
    write_recovery_receipt(nested_alias)
    with pytest.raises(ValueError, match="unexpected safety-like key"):
        formal.aggregate_shards(
            shard_roots,
            mechanics_root,
            output,
            freeze_manifest=tmp_path / "manifest.json",
            authorization_marker=tmp_path / "GO",
        )
    nested_false_alias = json.loads(json.dumps(original_receipt))
    nested_false_alias["rank_condition_precheck"]["training_performed"] = False
    with pytest.raises(ValueError, match="unexpected safety-like key"):
        formal._validate_receipt_safety_fields(
            nested_false_alias,
            "synthetic aggregate nested alias",
            receipt_kind="aggregate",
        )
    benign_scientific_names = json.loads(json.dumps(original_receipt))
    benign_scientific_names["rank_condition_precheck"].update(
        {"training_count": 92, "force_design_rank": 65, "train_gate": True}
    )
    formal._validate_receipt_safety_fields(
        benign_scientific_names,
        "synthetic aggregate scientific names",
        receipt_kind="aggregate",
    )
    nested_geometry_alias = json.loads(json.dumps(original_receipt))
    nested_geometry_alias["geometry_loader_isolation"]["training_performed"] = True
    write_recovery_receipt(nested_geometry_alias)
    with pytest.raises(ValueError, match="geometry_loader_isolation"):
        formal.aggregate_shards(
            shard_roots,
            mechanics_root,
            output,
            freeze_manifest=tmp_path / "manifest.json",
            authorization_marker=tmp_path / "GO",
        )
    numeric_false_tamper = json.loads(json.dumps(original_receipt))
    numeric_false_tamper["force_labels_used"] = 0
    write_recovery_receipt(numeric_false_tamper)
    with pytest.raises(ValueError, match="force_labels_used"):
        formal.aggregate_shards(
            shard_roots,
            mechanics_root,
            output,
            freeze_manifest=tmp_path / "manifest.json",
            authorization_marker=tmp_path / "GO",
        )
    numeric_geometry_tamper = json.loads(json.dumps(original_receipt))
    numeric_geometry_tamper["geometry_loader_isolation"][
        "force_labels_parsed"
    ] = 0
    write_recovery_receipt(numeric_geometry_tamper)
    with pytest.raises(ValueError, match="geometry_loader_isolation"):
        formal.aggregate_shards(
            shard_roots,
            mechanics_root,
            output,
            freeze_manifest=tmp_path / "manifest.json",
            authorization_marker=tmp_path / "GO",
        )
    write_recovery_receipt(original_receipt)

    self_consistent_status_tamper = json.loads(json.dumps(original_receipt))
    self_consistent_status_tamper["status"] = formal.STATUS_AGGREGATE_FAIL
    self_consistent_status_tamper["representation_precheck_pass"] = False
    self_consistent_status_tamper["train_geometry_and_rank1_gate"]["pass"] = False
    write_recovery_receipt(self_consistent_status_tamper)
    with pytest.raises(ValueError, match="recomputed status changed"):
        formal.aggregate_shards(
            shard_roots,
            mechanics_root,
            output,
            freeze_manifest=tmp_path / "manifest.json",
            authorization_marker=tmp_path / "GO",
        )
    write_recovery_receipt(original_receipt)

    with np.load(arrays_path, allow_pickle=False) as loaded_arrays:
        aggregate_arrays = {
            key: loaded_arrays[key].copy() for key in loaded_arrays.files
        }
    aggregate_arrays["thermal_b_A4"][0, 0] += 1.0e-7
    formal._atomic_npz(arrays_path, aggregate_arrays)
    self_consistent_array_hash_tamper = json.loads(json.dumps(original_receipt))
    self_consistent_array_hash_tamper["arrays_sha256"] = formal.sha256(arrays_path)
    write_recovery_receipt(self_consistent_array_hash_tamper)
    with pytest.raises(ValueError, match="arrays differ from current shard arrays"):
        formal.aggregate_shards(
            shard_roots,
            mechanics_root,
            output,
            freeze_manifest=tmp_path / "manifest.json",
            authorization_marker=tmp_path / "GO",
        )
    formal._atomic_bytes(arrays_path, original_arrays_bytes)
    write_recovery_receipt(original_receipt)

    stale = json.loads(receipt_path.read_text(encoding="utf-8"))
    stale["formal_contract_sha256"] = "0" * 64
    write_recovery_receipt(stale)
    with pytest.raises(ValueError, match="formal contract is stale"):
        formal.aggregate_shards(
            shard_roots,
            mechanics_root,
            output,
            freeze_manifest=tmp_path / "manifest.json",
            authorization_marker=tmp_path / "GO",
        )


def test_shard_atomic_artifacts_markers_and_completion_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The mocked design/query payload is intentionally smaller than production.
    monkeypatch.setattr(
        formal, "_validate_recursive_receipt_schema", lambda *_args, **_kwargs: None
    )
    thermal = _thermal_structures()
    harmonic = _structures(32, 128, formal.HARMONIC_CONFIG_TYPE)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    authorization = {
        "freeze_manifest_sha256": "m",
        "authorization_marker_sha256": "g",
        "formal_source_sha256": {
            name: formal.sha256(path) for name, path in formal.FORMAL_SOURCE_PATHS.items()
        },
    }
    monkeypatch.setattr(formal, "validate_execution_authorization", lambda *_args: authorization)
    monkeypatch.setattr(formal, "validate_frozen_sources", lambda: {"source": "hash"})
    monkeypatch.setattr(formal, "validate_inputs", lambda _inputs: {"input": "hash"})
    monkeypatch.setattr(formal, "expected_input_sha256", lambda: {"input": "hash"})
    monkeypatch.setattr(formal, "runtime_fingerprint", lambda _device: {"semantic_sha256": "env"})
    monkeypatch.setattr(formal, "load_endpoint", lambda *_args: object())
    monkeypatch.setattr(
        formal,
        "read_geometry_only_extxyz",
        lambda path: harmonic if "harmonic" in str(path) else thermal,
    )
    monkeypatch.setattr(formal, "read", lambda *_args, **_kwargs: thermal[0])
    monkeypatch.setattr(
        formal,
        "_design_record",
        lambda _model, _structure, _reference, _device: (_design_arrays(), _record()),
    )
    monkeypatch.setattr(
        formal,
        "_harmonic_record",
        lambda _model, _structure, _reference, _device: (
            {
                **{key: value for key, value in _design_arrays(128).items() if key in {"b_A4", "a", "c", "combined_energy_eV", "combined_force_eV_A"}},
                "fixed_carrier_energy_eV": np.zeros(1, dtype="<f8"),
                "fixed_carrier_force_eV_A": np.zeros((128, 3), dtype="<f8"),
            },
            _record(),
        ),
    )
    dummy = formal.FormalInputs(
        tmp_path / "endpoint.pt",
        tmp_path / "endpoint_receipt.json",
        tmp_path / "ENDPOINT_FROZEN",
        tmp_path / "reference_6x6.xyz",
        tmp_path / "reference_8x8.xyz",
        tmp_path / "train_thermal.xyz",
        tmp_path / "train_harmonic_lambda1_small_zero.xyz",
    )
    output = tmp_path / "shard"
    auth_args = {
        "freeze_manifest": tmp_path / "manifest.json",
        "authorization_marker": tmp_path / "GO",
    }
    result = formal.run_shard(dummy, output, 2, device="cpu", **auth_args)
    assert result["thermal_count"] == 30 and result["harmonic_count"] == 10
    assert (output / "THERMAL_FROZEN").is_file()
    assert (output / "HARMONIC_FROZEN").is_file()
    assert (output / "SENTINEL_FROZEN").is_file()
    loaded, arrays = formal._load_completed_shard(output)
    assert np.array_equal(arrays["thermal_global_index"], np.arange(2, 92, 3))
    recovered = formal.run_shard(dummy, output, 2, device="cpu", **auth_args)
    assert recovered["completion_recovered_without_recompute"] is True
    assert loaded["shard_id"] == 2
    (output / "FAILED").write_text("diagnostic\n", encoding="utf-8")
    with pytest.raises(ValueError, match="terminal markers"):
        formal._load_completed_shard(output)
    (output / "FAILED").unlink()
    receipt_path = output / "receipt.json"
    original = json.loads(receipt_path.read_text(encoding="utf-8"))

    def write_shard_receipt(candidate: dict) -> None:
        formal._atomic_json(receipt_path, candidate)
        (output / "DONE").write_text(
            formal.STATUS_SHARD + "\n" + formal.sha256(receipt_path) + "\n",
            encoding="ascii",
        )

    for safety_field in (
        "force_labels_used",
        "energy_labels_used",
        "can_authorize_fit_or_training",
    ):
        tampered = json.loads(json.dumps(original))
        tampered[safety_field] = True
        write_shard_receipt(tampered)
        with pytest.raises(ValueError, match="safety field changed"):
            formal._load_completed_shard(output)
    for geometry_field in formal.GEOMETRY_LOADER_ISOLATION:
        tampered = json.loads(json.dumps(original))
        tampered["geometry_loader_isolation"][geometry_field] = True
        write_shard_receipt(tampered)
        with pytest.raises(ValueError, match="geometry_loader_isolation"):
            formal._load_completed_shard(output)
    extra_top_level = json.loads(json.dumps(original))
    extra_top_level["fit_performed"] = True
    write_shard_receipt(extra_top_level)
    with pytest.raises(ValueError, match="top-level receipt schema changed"):
        formal._load_completed_shard(output)
    nested_alias = json.loads(json.dumps(original))
    nested_alias["thermal_receipts"][0]["training_performed"] = True
    write_shard_receipt(nested_alias)
    with pytest.raises(ValueError, match="unexpected safety-like key"):
        formal._load_completed_shard(output)
    nested_geometry_alias = json.loads(json.dumps(original))
    nested_geometry_alias["geometry_loader_isolation"]["training_performed"] = True
    write_shard_receipt(nested_geometry_alias)
    with pytest.raises(ValueError, match="geometry_loader_isolation"):
        formal._load_completed_shard(output)
    write_shard_receipt(original)

    stale = json.loads(receipt_path.read_text(encoding="utf-8"))
    stale["formal_contract_sha256"] = "0" * 64
    formal._atomic_json(receipt_path, stale)
    (output / "DONE").write_text(
        formal.STATUS_SHARD + "\n" + formal.sha256(receipt_path) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="formal contract is stale"):
        formal._load_completed_shard(output)


def test_mechanics_completed_recovery_rejects_self_consistent_safety_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Full production mechanics has a 1,815-path schema; this loader fixture
    # isolates array/recovery semantics while structural mutation is tested below.
    monkeypatch.setattr(
        formal, "_validate_recursive_receipt_schema", lambda *_args, **_kwargs: None
    )
    authorization = {
        "formal_source_sha256": {
            name: formal.sha256(path)
            for name, path in formal.FORMAL_SOURCE_PATHS.items()
        }
    }
    monkeypatch.setattr(formal, "validate_frozen_sources", lambda: {"source": "hash"})
    monkeypatch.setattr(formal, "expected_input_sha256", lambda: {"input": "hash"})
    output = tmp_path / "mechanics"
    output.mkdir()
    arrays = _passing_mechanics_arrays()
    arrays_path = output / "mechanics_arrays.npz"
    formal._atomic_npz(arrays_path, arrays)
    receipt = {
        **_passing_mechanics_receipt(),
        "format": formal.MECHANICS_FORMAT,
        "status": formal.STATUS_MECHANICS,
        "formal_contract_sha256": formal.FORMAL_CONTRACT_SHA256,
        "execution_authorization": authorization,
        "frozen_source_sha256": {"source": "hash"},
        "input_sha256": {"input": "hash"},
        "runtime_fingerprint": {"semantic_sha256": "mechanics-env"},
        "full_H_stage_receipts": [],
        "cuda_peak_allocated_bytes": 0,
        "dtype": "torch.float64",
        "arrays_sha256": formal.sha256(arrays_path),
        "array_schema": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "elapsed_seconds": 1.0,
    }
    receipt_path = output / "receipt.json"

    def write_mechanics_receipt(candidate: dict) -> None:
        formal._atomic_json(receipt_path, candidate)
        formal._atomic_bytes(output / "EXIT_CODE", b"0\n")
        formal._atomic_bytes(
            output / "DONE",
            (
                formal.STATUS_MECHANICS
                + "\n"
                + formal.sha256(receipt_path)
                + "\n"
            ).encode("ascii"),
        )

    write_mechanics_receipt(receipt)
    formal._load_completed_mechanics(output)
    for safety_field in (
        "force_labels_used",
        "energy_labels_used",
        "can_authorize_fit_or_training",
    ):
        tampered = json.loads(json.dumps(receipt))
        tampered[safety_field] = True
        write_mechanics_receipt(tampered)
        with pytest.raises(ValueError, match="safety field changed"):
            formal._load_completed_mechanics(output)
    for geometry_field in formal.GEOMETRY_LOADER_ISOLATION:
        tampered = json.loads(json.dumps(receipt))
        tampered["geometry_loader_isolation"][geometry_field] = True
        write_mechanics_receipt(tampered)
        with pytest.raises(ValueError, match="geometry_loader_isolation"):
            formal._load_completed_mechanics(output)
    extra_top_level = json.loads(json.dumps(receipt))
    extra_top_level["fit_or_training"] = True
    write_mechanics_receipt(extra_top_level)
    with pytest.raises(ValueError, match="top-level receipt schema changed"):
        formal._load_completed_mechanics(output)
    nested_alias = json.loads(json.dumps(receipt))
    nested_alias["mechanics"]["training_performed"] = True
    write_mechanics_receipt(nested_alias)
    with pytest.raises(ValueError, match="unexpected safety-like key"):
        formal._load_completed_mechanics(output)
    nested_geometry_alias = json.loads(json.dumps(receipt))
    nested_geometry_alias["geometry_loader_isolation"]["training_performed"] = True
    write_mechanics_receipt(nested_geometry_alias)
    with pytest.raises(ValueError, match="geometry_loader_isolation"):
        formal._load_completed_mechanics(output)


def test_launcher_plan_cannot_execute() -> None:
    import launch_graphene_r2r0_formal as launcher

    assert launcher.PLAN["remote_execution_authorized"] == "only_by_valid_external_GO_marker"
    assert set(launcher.ALLOWED_TARGETS) == {
        "root@100.80.236.112",
        "root@100.123.220.57",
        "howardwang@100.105.21.7",
    }
    command_a = launcher._remote_run_command("shard0", "a" * 12, 3)
    command_rtx = launcher._remote_run_command("shard2", "a" * 12, 3)
    assert command_a[:2] == ["ssh", "-o"]
    assert "root@100.80.236.112" in command_a
    assert "/root/miniconda3/bin/conda" in " ".join(command_a)
    assert "/home/howardwang/miniconda3/bin/conda" in " ".join(command_rtx)
    assert "attempt_0003" in " ".join(command_a)
    assert not any("remote_root" in value for value in command_a)
    with pytest.raises(ValueError, match="attempt 3"):
        launcher._remote_run_command("shard0", "a" * 12, 2)


def test_launcher_remote_symlink_guard_shell_render_is_executable(tmp_path: Path) -> None:
    import launch_graphene_r2r0_formal as launcher

    ordinary = tmp_path / "ordinary" / "future"
    rendered = launcher._ssh(
        "root@100.80.236.112", "python3", "-c", launcher.REMOTE_SYMLINK_GUARD, str(ordinary)
    )[-1]
    assert subprocess.run(["/bin/sh", "-c", rendered], check=False).returncode == 0
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing")
    rendered_broken = launcher._ssh(
        "root@100.80.236.112", "python3", "-c", launcher.REMOTE_SYMLINK_GUARD, str(broken / "child")
    )[-1]
    assert subprocess.run(["/bin/sh", "-c", rendered_broken], check=False).returncode != 0
    fresh_root = tmp_path / "fresh_bundle"
    rendered_fresh = launcher._ssh(
        "root@100.80.236.112",
        "python3",
        "-c",
        launcher.REMOTE_SYMLINK_GUARD,
        str(fresh_root),
        "require-absent",
    )[-1]
    assert subprocess.run(["/bin/sh", "-c", rendered_fresh], check=False).returncode == 0
    stale_root = tmp_path / "stale_bundle"
    stale_root.mkdir()
    rendered_stale = launcher._ssh(
        "root@100.80.236.112",
        "python3",
        "-c",
        launcher.REMOTE_SYMLINK_GUARD,
        str(stale_root),
        "require-absent",
    )[-1]
    assert subprocess.run(["/bin/sh", "-c", rendered_stale], check=False).returncode != 0


def test_launcher_staging_uses_only_allowlist_and_verifies_every_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import launch_graphene_r2r0_formal as launcher

    manifest = tmp_path / "freeze_manifest.json"
    formal.prepare_freeze_manifest(manifest)
    marker = tmp_path / "R2R0_FORMAL_GO"
    marker.write_text(formal.sha256(manifest) + "\n", encoding="ascii")
    calls = []

    def fake_run(command, **kwargs):
        command = list(command)
        calls.append(command)
        stdout_content = ""
        remote_argv = shlex.split(command[-1]) if command[0] == "ssh" else []
        if "sha256sum" in remote_argv:
            remote_path = remote_argv[-1]
            if remote_path.endswith("/freeze_manifest.json"):
                digest = formal.sha256(manifest)
            elif remote_path.endswith("/R2R0_FORMAL_GO"):
                digest = formal.sha256(marker)
            else:
                relative = None
                for spec in launcher.NODE_SPECS.values():
                    prefix = launcher._bundle_root(spec, "a" * 12) + "/"
                    if remote_path.startswith(prefix):
                        relative = remote_path.removeprefix(prefix)
                        break
                assert relative is not None
                digest = formal.sha256(ROOT / relative)
            stdout_content = digest + "  " + remote_path + "\n"
        if "stdout" in kwargs:
            kwargs["stdout"].write(stdout_content.encode("utf-8"))
        if "stderr" in kwargs:
            kwargs["stderr"].write(b"")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    logs = tmp_path / "logs"
    logs.mkdir()
    receipts = launcher._stage_and_verify(
        manifest, marker, logs, "a" * 12, 3, resume=False
    )
    assert len(receipts) == 203
    assert len(list(logs.iterdir())) == 609
    assert all(
        receipt["stdout_basename"]
        and receipt["stderr_basename"]
        and receipt["stdout_sha256"]
        and receipt["stderr_sha256"]
        for receipt in receipts
    )
    assert any("graphene_r2o_taylor_null.py" in " ".join(call) for call in calls)
    assert sum("endpoint.pt" in " ".join(call) and call[0] == "scp" for call in calls) == 3
    fresh_bundle_guards = [
        call
        for call in calls
        if call[0] == "ssh" and "require-absent" in shlex.split(call[-1])
    ]
    assert len(fresh_bundle_guards) == 3
    assert {
        shlex.split(call[-1])[-2] for call in fresh_bundle_guards
    } == {
        launcher._bundle_root(spec, "a" * 12)
        for spec in launcher.NODE_SPECS.values()
    }
    for call in calls:
        targets = [item for item in call if "@100." in item]
        assert all(
            target.split(":", 1)[0] in launcher.ALLOWED_TARGETS for target in targets
        )
        if call[0] in {"ssh", "scp"}:
            assert "BatchMode=yes" in call and "ConnectTimeout=20" in call


def test_launcher_success_inventory_is_exactly_220_commands_660_logs(
    tmp_path: Path,
) -> None:
    import launch_graphene_r2r0_formal as launcher

    logs = tmp_path / "logs"
    logs.mkdir()
    for index in range(launcher.EXPECTED_SUCCESSFUL_COMMAND_COUNT):
        name = f"command_{index:03d}"
        stdout = logs / f"{name}.stdout"
        stderr = logs / f"{name}.stderr"
        command_path = logs / f"{name}.command.json"
        stdout.write_bytes(f"stdout-{index}\n".encode("ascii"))
        stderr.write_bytes(b"")
        record = launcher._command_receipt(["synthetic-command", str(index)])
        record.update(
            {
                "returncode": 0,
                "stdout_basename": stdout.name,
                "stderr_basename": stderr.name,
                "stdout_sha256": formal.sha256(stdout),
                "stderr_sha256": formal.sha256(stderr),
                "command_basename": command_path.name,
                "expected_sha256": None,
                "observed_sha256": None,
            }
        )
        formal._atomic_json(command_path, record)
    assert len(list(logs.iterdir())) == launcher.EXPECTED_SUCCESSFUL_LOG_FILE_COUNT
    assert len(launcher._validate_command_log_inventory(logs)) == 220

    stream = logs / "command_000.stdout"
    original_stream = stream.read_bytes()
    stream.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="stream hash"):
        launcher._validate_command_log_inventory(logs)
    stream.write_bytes(original_stream)

    command_path = logs / "command_000.command.json"
    original_command = command_path.read_bytes()
    changed = json.loads(original_command)
    changed["stdout_basename"] = None
    formal._atomic_json(command_path, changed)
    with pytest.raises(ValueError, match="basename"):
        launcher._validate_command_log_inventory(logs)
    command_path.write_bytes(original_command)

    orphan = logs / "orphan"
    orphan.write_bytes(b"orphan")
    with pytest.raises(ValueError, match="inventory"):
        launcher._validate_command_log_inventory(logs)
    orphan.unlink()

    command_path.unlink()
    with pytest.raises(ValueError, match="command count"):
        launcher._validate_command_log_inventory(logs)
    command_path.write_bytes(original_command)

    changed = json.loads(original_command)
    changed["returncode"] = 1
    formal._atomic_json(command_path, changed)
    with pytest.raises(ValueError, match="nonzero"):
        launcher._validate_command_log_inventory(logs)


def test_launcher_waits_for_all_shards_before_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import launch_graphene_r2r0_formal as launcher

    authorization = {
        "freeze_manifest_sha256": "c" * 64,
        "authorization_marker_sha256": "d" * 64,
        "formal_source_sha256": {},
    }
    monkeypatch.setattr(launcher, "validate_execution_authorization", lambda *_args: authorization)
    monkeypatch.setattr(launcher, "_stage_and_verify", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(launcher, "_run_logged", lambda command, _logs, _name: launcher._command_receipt(command))
    waited = []
    codes = iter((1, 0, 0))

    class FakeProcess:
        def __init__(self, _command, **_kwargs):
            self.code = next(codes)

        def wait(self):
            waited.append(self.code)
            return self.code

    monkeypatch.setattr(launcher.subprocess, "Popen", FakeProcess)
    output = tmp_path / "launch"
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "GO").write_text("go\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="all children joined"):
        launcher.execute(
            freeze_manifest=tmp_path / "manifest.json",
            authorization_marker=tmp_path / "GO",
            collect_root=output,
            attempt=3,
        )
    assert waited == [1, 0, 0]
    assert (output / "FAILED").is_file() and not (output / "DONE").exists()


def test_launcher_propagates_aggregate_scientific_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import launch_graphene_r2r0_formal as launcher

    authorization = {
        "freeze_manifest_sha256": "e" * 64,
        "authorization_marker_sha256": "f" * 64,
        "formal_source_sha256": {
            name: formal.sha256(path)
            for name, path in formal.FORMAL_SOURCE_PATHS.items()
        },
    }
    monkeypatch.setattr(launcher, "validate_execution_authorization", lambda *_args: authorization)
    monkeypatch.setattr(launcher, "_stage_and_verify", lambda *_args, **_kwargs: [])

    class FakeProcess:
        def __init__(self, _command, **_kwargs):
            pass

        def wait(self):
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", FakeProcess)
    logged = []

    def fake_logged(command, _logs, _name):
        command = list(command)
        if command[0] == "scp":
            destination = Path(command[-1])
            destination.mkdir()
            (destination / "receipt.json").write_text("{}\n", encoding="utf-8")
            (destination / "DONE").write_text("done\n", encoding="ascii")
            (destination / "EXIT_CODE").write_text("0\n", encoding="ascii")
            if destination.name.startswith("shard"):
                for role in ("THERMAL", "HARMONIC", "SENTINEL"):
                    (destination / f"{role}_FROZEN").write_text(
                        "frozen\n", encoding="ascii"
                    )
                    (destination / f"{role.lower()}_arrays.npz").write_bytes(
                        b"arrays"
                    )
            elif destination.name == "mechanics":
                (destination / "mechanics_arrays.npz").write_bytes(b"arrays")
        if command[0] == "/Users/howardwang/miniconda3/bin/conda":
            destination = Path(command[command.index("--output") + 1])
            destination.mkdir()
            (destination / "arrays.npz").write_bytes(b"arrays")
            aggregate = {
                key: None
                for key in formal.RECEIPT_TOP_LEVEL_KEYS["aggregate"]
            }
            aggregate.update({
                "format": formal.AGGREGATE_FORMAT,
                "status": formal.STATUS_AGGREGATE_INCONCLUSIVE,
                "formal_contract_sha256": formal.FORMAL_CONTRACT_SHA256,
                "execution_authorization": authorization,
                "frozen_source_sha256": {},
                "frozen_R2R_canonical_sha256": formal.FROZEN_R2R_CANONICAL_SHA256,
                "shard_receipt_sha256": [],
                "shard_root_manifest_sha256": [],
                "input_sha256": {},
                "mechanics_receipt_sha256": "synthetic",
                "mechanics_root_manifest_sha256": "synthetic",
                "array_schema": {},
                "arrays_sha256": formal.sha256(destination / "arrays.npz"),
                "representation_precheck_pass": False,
                "numerically_inconclusive": True,
                **formal.AGGREGATE_SAFETY_FIELDS,
            })
            formal._atomic_json(destination / "receipt.json", aggregate)
            (destination / "DONE").write_text("done\n", encoding="ascii")
            (destination / "EXIT_CODE").write_text("0\n", encoding="ascii")
        record = launcher._command_receipt(command)
        record.update(
            {
                "returncode": 0,
                "stdout_basename": f"{_name}.stdout",
                "stderr_basename": f"{_name}.stderr",
                "stdout_sha256": "0" * 64,
                "stderr_sha256": "0" * 64,
                "command_basename": f"{_name}.command.json",
                "expected_sha256": None,
                "observed_sha256": None,
            }
        )
        logged.append(record)
        return record

    monkeypatch.setattr(launcher, "_run_logged", fake_logged)
    monkeypatch.setattr(
        launcher,
        "_validate_receipt_contract",
        lambda receipt, label, *, receipt_kind: formal._validate_receipt_safety_fields(
            receipt, label, receipt_kind=receipt_kind
        ),
    )
    monkeypatch.setattr(
        launcher, "_validate_command_log_inventory", lambda _logs: list(logged)
    )
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "GO").write_text("go\n", encoding="ascii")
    receipt = launcher.execute(
        freeze_manifest=tmp_path / "manifest.json",
        authorization_marker=tmp_path / "GO",
        collect_root=tmp_path / "launch",
        attempt=3,
    )
    assert receipt["status"] == formal.STATUS_AGGREGATE_INCONCLUSIVE
    assert receipt["aggregate_receipt"]["numerically_inconclusive"] is True
    assert (tmp_path / "launch" / "DONE").read_text().splitlines()[0] == formal.STATUS_AGGREGATE_INCONCLUSIVE
    recovered = launcher.execute(
        freeze_manifest=tmp_path / "manifest.json",
        authorization_marker=tmp_path / "GO",
        collect_root=tmp_path / "launch",
        attempt=3,
    )
    assert recovered["completion_recovered_without_recompute"] is True
    launch_root = tmp_path / "launch"
    launch_receipt_path = launch_root / "launch_receipt.json"
    original_launch_receipt = json.loads(
        launch_receipt_path.read_text(encoding="utf-8")
    )
    for safety_field, invalid_value in (
        ("fit_or_training", True),
        ("fit_or_training", 0),
        ("held_or_support_access", True),
        ("held_or_support_access", 0),
    ):
        tampered = json.loads(json.dumps(original_launch_receipt))
        tampered[safety_field] = invalid_value
        formal._atomic_json(launch_receipt_path, tampered)
        (launch_root / "DONE").write_text(
            tampered["status"]
            + "\n"
            + formal.sha256(launch_receipt_path)
            + "\n",
            encoding="ascii",
        )
        with pytest.raises(ValueError, match="launcher recovery safety field"):
            launcher.execute(
                freeze_manifest=tmp_path / "manifest.json",
                authorization_marker=tmp_path / "GO",
                collect_root=launch_root,
                attempt=3,
            )
    for alias in ("training_performed", "force_labels_used"):
        tampered = json.loads(json.dumps(original_launch_receipt))
        tampered[alias] = True
        formal._atomic_json(launch_receipt_path, tampered)
        (launch_root / "DONE").write_text(
            tampered["status"]
            + "\n"
            + formal.sha256(launch_receipt_path)
            + "\n",
            encoding="ascii",
        )
        with pytest.raises(
            ValueError, match=r"recursive (?:key-path )?schema"
        ):
            launcher.execute(
                freeze_manifest=tmp_path / "manifest.json",
                authorization_marker=tmp_path / "GO",
                collect_root=launch_root,
                attempt=3,
            )
    nested_alias = json.loads(json.dumps(original_launch_receipt))
    nested_alias["aggregate_receipt"]["training_performed"] = True
    formal._atomic_json(launch_receipt_path, nested_alias)
    (launch_root / "DONE").write_text(
        nested_alias["status"]
        + "\n"
        + formal.sha256(launch_receipt_path)
        + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match=r"recursive (?:key-path )?schema"):
        launcher.execute(
            freeze_manifest=tmp_path / "manifest.json",
            authorization_marker=tmp_path / "GO",
            collect_root=launch_root,
            attempt=3,
        )
    for alias, value in SAFETY_ALIAS_ATTACKS:
        recursive_alias = json.loads(json.dumps(original_launch_receipt))
        recursive_alias["commands"][0][alias] = value
        formal._atomic_json(launch_receipt_path, recursive_alias)
        (launch_root / "DONE").write_text(
            recursive_alias["status"]
            + "\n"
            + formal.sha256(launch_receipt_path)
            + "\n",
            encoding="ascii",
        )
        with pytest.raises(ValueError, match=r"recursive (?:key-path )?schema"):
            launcher.execute(
                freeze_manifest=tmp_path / "manifest.json",
                authorization_marker=tmp_path / "GO",
                collect_root=launch_root,
                attempt=3,
            )
    benign_but_structurally_undeclared = json.loads(
        json.dumps(original_launch_receipt)
    )
    benign_but_structurally_undeclared["commands"][0].update(
        {"training_count": 92, "force_design_rank": 65, "train_gate": True}
    )
    with pytest.raises(ValueError, match=r"recursive (?:key-path )?schema"):
        launcher._validate_launch_safety_fields(benign_but_structurally_undeclared)
    formal._atomic_json(launch_receipt_path, original_launch_receipt)
    (launch_root / "DONE").write_text(
        original_launch_receipt["status"]
        + "\n"
        + formal.sha256(launch_receipt_path)
        + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="attempt 3"):
        launcher.execute(
            freeze_manifest=tmp_path / "manifest.json",
            authorization_marker=tmp_path / "GO",
            collect_root=tmp_path / "launch",
            attempt=4,
        )
    (tmp_path / "launch" / "FAILED").write_text("failure\n", encoding="utf-8")
    with pytest.raises(ValueError, match="terminal XOR"):
        launcher.execute(
            freeze_manifest=tmp_path / "manifest.json",
            authorization_marker=tmp_path / "GO",
            collect_root=tmp_path / "launch",
            attempt=3,
        )
