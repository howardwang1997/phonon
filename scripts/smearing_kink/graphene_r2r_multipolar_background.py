#!/usr/bin/env python3
"""R2R-0 zero-parameter multipolar-background design primitives.

This module is deliberately an implementation/preflight layer.  It has no
trainable parameters, never loads data by itself, and does not fit a readout.
The only public linear object is a 65-column Taylor-null force design matrix.
Later stages may fit coefficients only after a separate reviewed contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from mace.modules.utils import get_edge_vectors_and_lengths

from graphene_r2o_taylor_null import (
    Assignment,
    adapt_reference_cell,
    fixed_reference_graph,
    model_dtype,
    reordered_structure,
    solve_assignment,
    state_dict_sha256,
    validate_current_edge_set,
    validate_mace_architecture,
)


FORMAT = "graphene_r2r0_multipolar_background_v4_covariant_rigid_probe"
R2Q_ENDPOINT_STATE_SHA256 = (
    "733e223805ed3cbc20b645615a8edbd8d60f3cd25898d7cf629848b0aee29d47"
)
REFERENCE_CUTOFF_A = 6.0
REFERENCE_WEIGHT_DEGREE = 5
EXPECTED_NEIGHBORS_EXCLUDING_CENTER = 39
BACKGROUND_EFFECTIVE_RADIUS_A = 6.4
BACKGROUND_INTERACTION_DIAMETER_A = 12.8
MULTIPOLAR_BETA_A4 = 2.5e-8
RANK1_NUMERICAL_B_GATE_A4 = 1.0e-20
RANK1_NUMERICAL_A_GATE = 1.0e-12
TRACE_SCALE_A2 = 0.015
SIGNED_L0_CHANNELS = 32
LINEAR_DESIGN_WIDTH = 65
ZERO_COLUMN_RELATIVE_RMS = 1.0e-12
SCALED_CONDITION_REPORT_LIMIT = 1.0e8
EXPECTED_SPLIT_COUNTS = {"thermal92": 92}
MECHANICS_PROBE_COEFFICIENTS = tuple(
    float(value) for value in np.linspace(-0.2, 0.3, 65, dtype=np.float64)
)
AFFINE_COMPONENT_NAMES = (
    ("fixed_carrier",)
    + tuple(f"w0[{index}]" for index in range(32))
    + ("b1",)
    + tuple(f"w1[{index}]" for index in range(32))
)

EXPECTED_INPUTS = {
    "endpoint_checkpoint": {
        "basename": "endpoint.pt",
        "sha256": "31dd053a01c700ee73eedff35e85c02cf7765de2299ff418259c270b3f823eb5",
    },
    "endpoint_receipt": {
        "basename": "endpoint_receipt.json",
        "sha256": "810a45ab66a885a940e086a7be9c4304050a0772d477ecd32bd03c20c8ef43a1",
    },
    "endpoint_marker": {
        "basename": "ENDPOINT_FROZEN",
        "sha256": "5fa7771f4c8f7d32990e4bb0d0350e8f1f020c1685030626ac8555b06a4ce0ed",
    },
    "reference_6x6": {
        "basename": "reference_6x6.xyz",
        "sha256": "2793b514768c6d875831a1b62fcb2aa5f858830cb2de7b1b2d5492dabeb8da90",
    },
    "reference_8x8": {
        "basename": "reference_8x8.xyz",
        "sha256": "1c438550cb03d94cc80ca16e1ca89d24f87db4d732743b6c07128d47edf5b900",
    },
    "thermal92": {
        "basename": "train_thermal.xyz",
        "sha256": "cc2e9c68d418fba996e8ac89c8156c8adf00768406f6171aaa57f0a9bd2720c1",
    },
    "harmonic_zero32": {
        "basename": "train_harmonic_lambda1_small_zero.xyz",
        "sha256": "600c737b11987fb976724dd86042d384fc1abef9f2340ffa5c30d5fe26a8afdf",
    },
}

EXPECTED_REFERENCE_SEMANTIC_SHA256 = {
    "reference_6x6": "2d9d97aa3a994f1bc4db0965a1837d46269b0f588fc19d8c96308848c58ef6ea",
    "reference_8x8": "c433781cb5b3270dcd6f78f1ef7af0a42f978a67219416dd15051795aede431e",
}
EXPECTED_BASELINE_FORMAL_GRAPH_SHA256 = {
    "reference_6x6": "3a0c26908a936042bda6f7dcaf7b2520e9a9f14373c8c61561bd6040c552256e",
    "reference_8x8": "4728c7542ea0a89e3f0ca451f2bd5c87595207086e3a4f6e26c099571ff239ba",
}

FORBIDDEN_PATH_TOKENS = (
    "seed1",
    "seed2",
    "small12",
    "support",
    "reserved",
    "outer_fold",
    "held",
    "valid_e50",
    "small_gate",
    "full_report",
)

# Both scales were frozen from geometry-only statistics on the allowed 92+32
# train structures.  No force label enters either derivation.
SCALE_DERIVATION = {
    "input_geometry_sha256": {
        "thermal92": EXPECTED_INPUTS["thermal92"]["sha256"],
        "harmonic_zero32": EXPECTED_INPUTS["harmonic_zero32"]["sha256"],
    },
    "covariance_definition": (
        "samples are center u_i with weight 1 and neighbor u_j with weight w_ij; "
        "mu_i is their normalized weighted mean and C_i is their normalized "
        "weighted covariance"
    ),
    "trace_scale": {
        "thermal92_nodewise_median_A2": 0.014458039461185433,
        "rounding_rule": "nearest multiple of 0.005 A2, ties-to-even",
        "frozen_s0_A2": TRACE_SCALE_A2,
    },
    "multipolar_scale": {
        "thermal92_min_cross_product_square_A4": 1.8338961456148413e-6,
        "harmonic_zero32_direct_pair_max_cross_product_square_A4": 2.2896187649234325e-34,
        "rank1_numerical_gate_A4": RANK1_NUMERICAL_B_GATE_A4,
        "candidate_grid_A4": "{1,2.5,5}*10^n",
        "rule": (
            "largest grid value with min_thermal_b/beta >= 50; "
            "harmonic_zero32 is checked as numerical rank-1"
        ),
        "frozen_beta_A4": MULTIPOLAR_BETA_A4,
    },
}


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def semantic_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def reference_semantic_payload(reference: Atoms) -> dict:
    """Rotation-invariant ordered-tiling fingerprint for a frozen reference."""
    cell = np.asarray(reference.cell, dtype=np.float64)
    fractional = np.asarray(reference.get_scaled_positions(wrap=False), dtype=np.float64)
    periodic = np.asarray(reference.pbc, dtype=bool)
    for axis, is_periodic in enumerate(periodic):
        if is_periodic:
            fractional[:, axis] -= np.floor(fractional[:, axis])
            fractional[np.isclose(fractional[:, axis], 1.0, atol=5.0e-11), axis] = 0.0
    rounded_fractional = np.round(fractional, 10)
    rounded_cell_metric = np.round(cell @ cell.T, 10)
    # JSON distinguishes -0.0 from +0.0 even though IEEE comparisons do not.
    # BLAS-dependent rigid transforms may emit either sign for exact zeros.
    rounded_fractional[rounded_fractional == 0.0] = 0.0
    rounded_cell_metric[rounded_cell_metric == 0.0] = 0.0
    return {
        "numbers": np.asarray(reference.numbers, dtype=int).tolist(),
        "pbc": periodic.tolist(),
        "fractional_positions_ordered": rounded_fractional.tolist(),
        "cell_metric_A2": rounded_cell_metric.tolist(),
    }


def reference_semantic_sha256(reference: Atoms) -> str:
    return semantic_sha256(reference_semantic_payload(reference))


def validate_reference_semantics(reference: Atoms) -> str:
    role = {72: "reference_6x6", 128: "reference_8x8"}.get(len(reference))
    if role is None:
        raise ValueError("R2R-0 supports only frozen 6x6/8x8 references")
    observed = reference_semantic_sha256(reference)
    if observed != EXPECTED_REFERENCE_SEMANTIC_SHA256[role]:
        raise ValueError("R2R reference ordered tiling/background origin changed")
    return observed


SCALE_DERIVATION_SHA256 = semantic_sha256(SCALE_DERIVATION)
MECHANICS_PROBE_COEFFICIENTS_SHA256 = semantic_sha256(
    list(MECHANICS_PROBE_COEFFICIENTS)
)
AFFINE_COMPONENT_NAMES_SHA256 = semantic_sha256(list(AFFINE_COMPONENT_NAMES))
MECHANICS_AFFINE_COEFFICIENTS_SHA256 = semantic_sha256(
    [1.0, *MECHANICS_PROBE_COEFFICIENTS]
)
FIXED_CARRIER_COEFFICIENT_SHA256 = semantic_sha256([1.0])


CANONICAL_CONTRACT = {
    "format": FORMAT,
    "stage": "R2R-0_zero_parameter_review_candidate",
    "reference_semantic_signed_zero_policy": (
        "after rounding fractional positions and cell metric to 10 decimals, "
        "canonicalize every IEEE value equal to zero to +0.0 before JSON; "
        "nonzero reference tampering remains fail-closed"
    ),
    "authorization": {
        "may_fit": False,
        "may_train": False,
        "may_open_held": False,
        "may_launch_remote": False,
        "may_authorize_R2R-1": False,
    },
    "inputs": EXPECTED_INPUTS,
    "endpoint_model_state_sha256": R2Q_ENDPOINT_STATE_SHA256,
    "production_integration": {
        "only_public_formal_API": "production_linear_design_query",
        "canonical_combination_method": (
            "ProductionLinearDesignQuery.canonical_combined_probe"
        ),
        "complete_Hessian_API": "production_combined_energy_force_hessian",
        "fixed_carrier_parity_APIs": [
            "production_fixed_carrier_energy_force",
            "production_fixed_carrier_energy_force_hessian",
        ],
        "complete_Hessian_implementation": (
            "shared hash-bound context; corrected live-weighted per-node Taylor2 "
            "carrier plus 65 raw zero-2-jet parameter columns; one combined scalar "
            "energy, force and full Hessian"
        ),
        "linear_design_create_graph": False,
        "legacy_66_column_create_graph_mechanics_forbidden": True,
        "production_linear_implementation": (
            "one live-weighted node-energy Taylor2 carrier plus raw/Jacobian of "
            "65 zero-2-jet parameter columns"
        ),
        "manual_primitive_assembly_for_formal_forbidden": True,
        "float32_origin_graph_required": True,
        "baseline_formal_graph_sha256": EXPECTED_BASELINE_FORMAL_GRAPH_SHA256,
        "rotated_graph_policy": (
            "rigid_transform_probe first rebuilds and exact-hash verifies the frozen "
            "baseline 6x6/8x8 float32-origin formal graph; positions, shifts and cell "
            "are then right-multiplied by Q.T in model dtype, while edge_index, "
            "unit_shifts, node_attrs, batch, ptr, head and pbc remain byte exact"
        ),
        "rotated_background_policy": (
            "reuse the exact baseline 6 A receiver/sender/image topology, reference "
            "distances, quintic weights and normalization; live centered displacement "
            "q vectors transform with the synchronized live structure/reference Q"
        ),
        "native_rotated_graph_rebuild_role": (
            "explicit diagnostic receipt only; never the O3 physics-gate graph"
        ),
        "native_rebuild_physical_multiset_veto": (
            "only the exact directed physical edge identity-key set is a veto; native "
            "vector/length and background distance/weight/normalization/shortest-cell "
            "differences are recorded as diagnostic scalars and never threshold a gate"
        ),
        "rigid_transform_public_input_covariance": (
            "reference and live structure positions/cell must be array-exact to their "
            "hash-bound baseline templates right-multiplied by Q.T; internal reordered/"
            "adapted arrays use the frozen named machine-level absolute tolerance"
        ),
        "rigid_transform_internal_covariance_atol_A": 1.0e-12,
        "rigid_transform_probe_baseline_templates": {
            "structure": "thermal92 global index 0",
            "structure_full_semantic_sha256": (
                "d3f6eca52753a6407c58d107374e06e80cec470c3668c1c6f1f034ea59488766"
            ),
            "reference6_full_semantic_sha256": (
                "39cf74c68a5ce629ab708ce9158f49d72a7e955895d5b01fa10c2883caa861e7"
            ),
        },
        "rigid_transform_v4_CPU_regression": {
            "rng": 83,
            "proper": {
                "derived_graph_sha256": (
                    "8f40d3565ffbd0e58b5b40dd4a40f35165a2904822b950ef45e7ece3bbcf8f87"
                ),
                "native_rebuilt_graph_sha256": (
                    "436bc0b781316243c0489a26655eb3b023f5b7d9e6271cc1cffec6e3fe913666"
                ),
                "fixed_force_covariance_max_abs_eV_A": 3.680389326632394e-14,
                "parameter_force_covariance_max_abs_eV_A": 5.81756864903582e-14,
                "combined_force_covariance_max_abs_eV_A": 6.361577931102147e-14,
                "native_rebuild_combined_force_difference_eV_A": (
                    2.670098361579054e-5
                ),
            },
            "improper": {
                "derived_graph_sha256": (
                    "67c225b78993c5bb061ed99437e07c15d04741351cee7cb512a85679fd39f8cf"
                ),
                "native_rebuilt_graph_sha256": (
                    "8283f4fd3f52dfd621f9c3597bed5ebb70a382ec3c1c631b9acbd7ee77b75868"
                ),
                "fixed_force_covariance_max_abs_eV_A": 4.718447854656915e-14,
                "parameter_force_covariance_max_abs_eV_A": 6.501049698570682e-14,
                "combined_force_covariance_max_abs_eV_A": 8.482103908136196e-14,
                "native_rebuild_combined_force_difference_eV_A": (
                    1.9484727560720172e-5
                ),
            },
            "force_labels_used": False,
            "physics_thresholds_unchanged": True,
        },
        "mechanics_probe_coefficients": list(MECHANICS_PROBE_COEFFICIENTS),
        "mechanics_probe_coefficients_sha256": MECHANICS_PROBE_COEFFICIENTS_SHA256,
        "affine_component_names": list(AFFINE_COMPONENT_NAMES),
        "affine_component_names_sha256": AFFINE_COMPONENT_NAMES_SHA256,
        "mechanics_affine_[1,p]_sha256": MECHANICS_AFFINE_COEFFICIENTS_SHA256,
        "mechanics_evaluates": "actual fixed_offset + X @ p",
        "node_energy_parity_probes": {
            "reference_6x6_semantic_sha256": EXPECTED_REFERENCE_SEMANTIC_SHA256[
                "reference_6x6"
            ],
            "thermal92_E50_seed0_global_index": 0,
            "thermal92_E50_seed0_structure_semantic_sha256": (
                "e64c2c5cc7681f710f01a22e9a374cf900c435387de16edfbb2037b025d941b9"
            ),
        },
        "corrected_carrier_CPU_prototype_receipt": {
            "thermal0_carrier_and_frozen_tail_full_H_total_wall_seconds": (
                66.63337216712534
            ),
            "thermal0_corrected_carrier_full_H_wall_seconds": 39.96564812492579,
            "process_ru_maxrss_raw_macos_bytes": 1302216704,
            "synthetic_explicit_node_energy_force_Hessian_parity_max": (
                1.2e-15
            ),
            "force_labels_used": False,
            "formal_launch_authorized": False,
        },
        "optimized_parameter_design_CPU_benchmark": {
            "probe": "thermal92_E50_seed0_global_index0",
            "raw_65_forward_seconds": 0.084,
            "raw_65_force_Jacobian_seconds": 1.150,
            "total_seconds": 1.233,
            "shape": [216, 65],
            "process_ru_maxrss_approx_MB": 582,
            "formal_must_not_use_per_column_reference_HVP": True,
        },
    },
    "encoder_history": {
        "gradient_seen_structures": "thermal92 plus harmonic_zero32 only",
        "endpoint_only_once_after_freeze": (
            "E50 seed1 and actual small-H were evaluated once only after the R2Q "
            "endpoint freeze and never wrote back to parameters, directions, or selection"
        ),
        "never_opened": "seed2 and support",
        "R2R0_encoder_update": False,
    },
    "background": {
        "reference_graph": "fixed reference-order directed 6 A neighbors",
        "reference_semantic_sha256": EXPECTED_REFERENCE_SEMANTIC_SHA256,
        "neighbors_excluding_center": EXPECTED_NEIGHBORS_EXCLUDING_CENTER,
        "center_sample": (
            "included with pre-centering relative displacement zero and weight 1; "
            "after weighted-mean subtraction its vector is -mu_relative"
        ),
        "degree5_reverse_smootherstep_weight": (
            "w(x)=1-10*x^3+15*x^4-6*x^5 for x=r_ref/6<1; w=0 otherwise; "
            "this is distinct from the frozen R2Q MACE degree-7 PolynomialCutoff(p=5)"
        ),
        "normalization": "W_i=1+sum_j w_ij; q_i0=1/W_i and q_ij=w_ij/W_i",
        "covariance": (
            "mu_i=q_i0*u_i+sum_j q_ij*u_j; C_i=q_i0*(u_i-mu_i)^2+"
            "sum_j q_ij*(u_j-mu_i)^2"
        ),
        "b": (
            "sum over unordered sample pairs j<k (center included) of q_ij*q_ik*"
            "norm((u_j-mu_i) cross (u_k-mu_i))^2; no extra factor 1/2"
        ),
        "a": "-expm1(-b/(2.5e-8 A4))",
        "rank1_evaluation": (
            "harmonic_zero32 is evaluated with direct b/a and actual Taylor-remainder E/F "
            "in a separate receipt; it never enters design scaling, rank, OOF, or fitting"
        ),
        "s": "trace(C)",
        "c": "s/(s+0.015 A2)",
        "scale_derivation_sha256": SCALE_DERIVATION_SHA256,
        "effective_radius_A": BACKGROUND_EFFECTIVE_RADIUS_A,
        "interaction_diameter_A": BACKGROUND_INTERACTION_DIAMETER_A,
        "no_wrap": {
            "6x6_shortest_in_plane_translation_A": 14.76,
            "8x8_shortest_in_plane_translation_A": 19.68,
            "6x6_diameter_margin_A": 1.96,
            "8x8_diameter_margin_A": 6.88,
            "strict_diameter_lt_translation": True,
            "same_source_duplicate_periodic_image_within_6A": False,
        },
        "size_parity_interpretation": (
            "6x6/8x8 field parity is implementation diagnostic; the authoritative "
            "deployment gate is localized Taylor-remainder total E/max-F parity"
        ),
    },
    "linear_design": {
        "affine_formula": (
            "r_epsilon_i=T2null[epsilon_R2Q_i]; G0=sum_i a_i*r_epsilon_i; "
            "G_R2R=G0+sum_i a_i*(w0 dot phi_i+c_i*(b1+w1 dot phi_i))"
        ),
        "fixed_offset": (
            "G0=sum_i a_i*T2null[epsilon_R2Q_i] with immutable coefficient +1; "
            "implemented exactly as one live-a weighted reference Taylor jet"
        ),
        "parameter_columns": [
            "columns 0:32 = sum_i a_i*phi_i,k",
            "column 32 = sum_i a_i*c_i (the only bias)",
            "columns 33:65 = sum_i a_i*c_i*phi_i,k",
        ],
        "width": LINEAR_DESIGN_WIDTH,
        "initial_readout_coefficients": "all 65 zero; fixed offset remains present",
        "global_intercept": False,
        "combination_semantics": (
            "replacement: frozen foundation + corrected G_R2R + frozen q6; "
            "never add the old R2Q Taylor tail a second time"
        ),
        "zero_readout_semantics": (
            "when all a_i=1, linearity makes G0 exactly the frozen whole-R2Q Taylor2 "
            "tail; exact rank-1 a=0 returns the zero-tail baseline"
        ),
        "fixed_offset_semantics": "per-node Taylor2 before multiplication by live a",
        "rank0_zero_jet_optimization": {
            "proof": (
                "at aligned reference u=0, centered q vectors are O(u), each cross "
                "product is O(u^2), b and a are O(u^4), while c is O(u^2); every "
                "one of the 65 parameter components contains factor a, hence its "
                "value, gradient and Hessian vanish and Taylor2(column)=raw column"
            ),
            "scope": (
                "only 65 parameter columns a*phi, a*c and a*c*phi at the hash-bound "
                "aligned reference; it is forbidden for the fixed carrier"
            ),
            "formal_design": "raw 65-vector parameter energy plus position Jacobian",
            "slow_generic_Taylor2_role": "audit_only_not_formal",
            "audit_policy": (
                "direct-pair O(u^4) is the authorization proof; the numerical gate "
                "checks every global node a_i/Jacobian and every independent local "
                "center+39-neighbor value/Jacobian/full 120x120 Hessian"
            ),
            "all_node_local_receipt": (
                "exact FP64 q-byte signatures and all-node exact-once coverage; each "
                "node uses a fresh local graph and is released after its full Hessian"
            ),
            "routine_test_policy": (
                "small synthetic audit in routine pytest; real 6x6/8x8 all-node "
                "audits are explicit slow gates"
            ),
            "local_direct_numeric_receipts": {
                "reference_6x6": {
                    "node_count": 72,
                    "exact_q_signature_count": 71,
                    "q_signature_orbits_sha256": (
                        "19ca084d9f54c17864d059bffc2da71e290e87bd8bab9dbea4ac6395441d4440"
                    ),
                    "probe_nodes_sha256": (
                        "35013e78c8691a14ba643ae6e528e387a76a351ce2bf3ccec566101e9ab676a5"
                    ),
                    "production_local_source_order_sha256": (
                        "4f0fcf5acf9669588bf7438fdb9eaf0b00dcff5c62c528b8d7637ffac3b744db"
                    ),
                    "observed_wall_seconds": 0.9827806251123548,
                    "observed_process_ru_maxrss_raw": 592297984,
                },
                "reference_8x8": {
                    "node_count": 128,
                    "exact_q_signature_count": 123,
                    "q_signature_orbits_sha256": (
                        "b6cec9b3886a71c076c715daeaf2dbbc48a42986a7c0b76c0becc4a144a0fdb5"
                    ),
                    "probe_nodes_sha256": (
                        "5c3946ef8052e49b3193e9baa95fb44168a50ef1eb5bb96939848915f4da66b8"
                    ),
                    "production_local_source_order_sha256": (
                        "6e44497aa2b55fc433c8155ce08f9f9b2695d9adcf9b298046dfd291958ecc56"
                    ),
                    "observed_wall_seconds": 1.9984580830205232,
                    "observed_process_ru_maxrss_raw": 584417280,
                },
                "nonzero_probe_base_sha256": (
                    "5e9fdb9d11a6cd6573121a00bbc379056bf6bbe3c2c729935de9b8be5ae91d4d"
                ),
                "all_node_zero_value_Jacobian_local_full_Hessian_max": 0.0,
                "force_labels_used": False,
            },
        },
        "rejected_legacy_carrier": {
            "formula": "Taylor2null[sum_i a_i*epsilon_R2Q_i]",
            "reason": (
                "because a=O(u^4), its reference 2-jet is zero and thermal a=1 "
                "returns raw epsilon rather than the frozen R2Q Taylor tail"
            ),
            "negative_regression_probe": (
                "thermal92_E50_seed0_global_index0 semantic hash "
                "e64c2c5cc7681f710f01a22e9a374cf900c435387de16edfbb2037b025d941b9"
            ),
            "negative_regression_force_difference_max_abs_min_eV_A": 0.1,
            "force_labels_used": False,
            "production_forbidden": True,
        },
        "R2Q_node_energy": "ScaleShift per-node interaction energy",
        "public_state_guard": (
            "recompute full state_dict semantic SHA on every node-observable call; "
            "require endpoint state 733e2238...29d47"
        ),
        "R2Q_signed_l0": (
            "16 even-parity l=0 channels from each of two interaction products; "
            "any 0o scalar is rejected"
        ),
        "column_scaling": "train-force RMS only; do not subtract column means",
        "zero_column_relative_RMS_threshold": ZERO_COLUMN_RELATIVE_RMS,
        "zero_column_policy": "fail_closed_no_reduced_design",
        "scaled_condition_report_limit": SCALED_CONDITION_REPORT_LIMIT,
    },
    "future_readout_boundary": {
        "background_scale_sources": (
            "geometry-only thermal92 plus harmonic_zero32 rank audit; no force gap"
        ),
        "readout_fit_source_if_later_authorized": "thermal92 residual force gap only",
        "ridge_scope": "65 newly introduced residual coefficients only",
        "ridge_or_readout_fit_in_R2R0": False,
        "continuous_outer_folds": {
            "folds": 4,
            "each_holdout": {
                "E50_seed0": 5,
                "T300": 9,
                "T600": 9,
            },
        },
        "harmonic_zero32_role": (
            "analytic/numerical rank-1 null gate only; use actual direct b/a and Taylor "
            "remainder E/F, never hand-zeroed design rows"
        ),
        "OOF_interpretation": (
            "conditional linear-readout OOF only: the frozen R2Q encoder has seen "
            "the upstream training corpus, so this is not encoder-level independence"
        ),
    },
    "rank1_boundary": {
        "analytic": "any exactly rank-1 displacement field has b=a=0",
        "verified_in_R2R0": "harmonic_zero32 only",
        "held_small_set": "must remain unopened in R2R0",
        "legacy_full25": (
            "only its rank-1 subset can use analytic gate-off; the other 13 "
            "large-amplitude rank-3 structures are report-only and may have a>0"
        ),
    },
    "fixed_gates": {
        "runner_override_allowed": False,
        "thermal92": {"min_b_A4": 1.0e-6, "min_a": 0.99},
        "harmonic_zero32": {
            "max_b_A4": 1.0e-20,
            "max_a": 1.0e-12,
            "max_Taylor_remainder_energy_eV": 1.0e-10,
            "max_Taylor_remainder_force_eV_A": 1.0e-9,
        },
        "R2Q_node_parity": {
            "energy_sum_abs_eV": 1.0e-10,
            "position_gradient_max_abs_eV_A": 1.0e-9,
        },
        "corrected_carrier_vs_frozen_R2Q_parity": {
            "energy_abs_eV": 1.0e-10,
            "force_max_abs_eV_A": 1.0e-9,
            "Hessian_max_abs_eV_A2": 1.0e-7,
            "probes": "reference_6x6 and thermal92_E50_seed0_global_index0",
        },
        "parameter_rank0_zero_jet": {
            "nodewise_a_value_max_abs": 0.0,
            "nodewise_a_Jacobian_max_abs_A-1": 0.0,
            "nodewise_a_Hessian_max_abs_A-2": 0.0,
            "probes": "reference_6x6 and reference_8x8",
            "nonzero_local_production_value_abs": 1.0e-14,
            "nonzero_local_production_gradient_max_abs_A-1": 1.0e-12,
            "nonzero_probe_b_over_beta_abs_difference": 1.0e-12,
            "nonzero_probe_target_a": 0.6321205588285577,
        },
        "O3_proper_and_improper": {
            "energy_abs_eV": 1.0e-6,
            "force_covariance_max_abs_eV_A": 1.0e-5,
        },
        "translation": {
            "energy_abs_eV": 1.0e-9,
            "force_max_abs_eV_A": 1.0e-8,
        },
        "permutation": {
            "energy_abs_eV": 1.0e-6,
            "force_max_abs_eV_A": 1.0e-5,
        },
        "native_cell_wrap_order_MIC_combined_probe": {
            "energy_abs_eV": 1.0e-6,
            "force_max_abs_eV_A": 1.0e-5,
            "raw_65_column_difference": "diagnostic_only",
        },
        "finite_difference": {
            "force_abs_eV_A": 1.0e-5,
            "force_Hessian_abs_eV_A2": 1.0e-5,
        },
        "nonreference_complete_Hessian": {
            "antisymmetry_max_abs_eV_A2": 1.0e-7,
            "translation_ASR_max_abs_eV_A2": 1.0e-7,
        },
        "localized_6x6_to_8x8_full_remainder": {
            "total_energy_abs_eV": 1.0e-7,
            "max_force_abs_eV_A": 1.0e-5,
        },
        "reference_Taylor_remainder": {
            "energy_abs_eV": 1.0e-10,
            "force_max_abs_eV_A": 1.0e-9,
            "Hessian_max_abs_eV_A2": 1.0e-7,
            "Hessian_antisymmetry_max_abs_eV_A2": 1.0e-7,
            "Hessian_translation_ASR_max_abs_eV_A2": 1.0e-7,
            "Gamma_K_frequency_drift_cm-1": 2.0,
        },
        "thermal_force_design": {
            "columns": 65,
            "all_columns_active": True,
            "scaled_rank": 65,
            "scaled_condition_number_max": 1.0e8,
        },
        "quintic_cutoff_inside_limit": {
            "value_abs": 1.0e-12,
            "first_derivative_abs_A-1": 1.0e-11,
            "second_derivative_abs_A-2": 1.0e-10,
            "third_derivative_abs_A-3_min": 1.0e-6,
        },
    },
    "required_mechanics": [
        "O3_proper_and_improper",
        "translation",
        "permutation",
        "order_and_MIC",
        "6x6_to_8x8_localized",
        "rank1_gate_off",
        "Taylor_reference_E_F_H_null",
        "locality_radius_and_diameter",
        "R2Q_node_energy_value_and_position_gradient_parity",
    ],
}
CANONICAL_CONTRACT_SHA256 = semantic_sha256(CANONICAL_CONTRACT)


def quintic_reference_weight(distance_A: torch.Tensor) -> torch.Tensor:
    """Degree-5 reverse smootherstep on the detached reference distance."""
    if not distance_A.is_floating_point():
        raise TypeError("reference distances must be floating point")
    x = distance_A / REFERENCE_CUTOFF_A
    inside = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
    return torch.where(distance_A < REFERENCE_CUTOFF_A, inside, torch.zeros_like(x))


def quintic_inside_cutoff_metrics() -> dict[str, float]:
    """Autograd audit of the polynomial's inside limit at exactly 6 A."""
    radius = torch.tensor(
        [REFERENCE_CUTOFF_A], dtype=torch.float64, requires_grad=True
    )
    x = radius / REFERENCE_CUTOFF_A
    value = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
    first = torch.autograd.grad(value.sum(), radius, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), radius, create_graph=True)[0]
    third = torch.autograd.grad(second.sum(), radius)[0]
    return {
        "value": float(value.detach()),
        "first_derivative_A-1": float(first.detach()),
        "second_derivative_A-2": float(second.detach()),
        "third_derivative_A-3": float(third.detach()),
    }


@dataclass(frozen=True)
class ReferenceNeighborhood:
    receiver: torch.Tensor
    sender: torch.Tensor
    image_integer: torch.Tensor
    reference_distance_A: torch.Tensor
    weight: torch.Tensor
    normalization: torch.Tensor
    atom_count: int
    shortest_in_plane_translation_A: float

    def validate(self) -> None:
        edge_count = self.receiver.numel()
        if any(
            tensor.shape[0] != edge_count
            for tensor in (
                self.sender,
                self.image_integer,
                self.reference_distance_A,
                self.weight,
            )
        ):
            raise ValueError("reference-neighborhood arrays disagree")
        if self.image_integer.shape != (edge_count, 3):
            raise ValueError("reference image integers have wrong shape")
        degree = torch.bincount(self.receiver, minlength=self.atom_count)
        if not torch.all(degree == EXPECTED_NEIGHBORS_EXCLUDING_CENTER):
            raise ValueError("6 A graphene reference degree changed")
        for atom in range(self.atom_count):
            sender = self.sender[self.receiver == atom]
            if torch.unique(sender).numel() != sender.numel():
                raise ValueError("same source appears through duplicate periodic images")
        if torch.any(self.reference_distance_A >= REFERENCE_CUTOFF_A):
            raise ValueError("reference graph includes an edge at/outside 6 A")
        if torch.any(self.weight <= 0.0):
            raise ValueError("reference quintic weights must be strictly positive")
        if self.shortest_in_plane_translation_A <= BACKGROUND_INTERACTION_DIAMETER_A:
            raise ValueError("R2R background interaction diameter can wrap in-plane")
        expected_norm = torch.ones_like(self.normalization)
        expected_norm.index_add_(0, self.receiver, self.weight)
        tolerance = 16.0 * torch.finfo(self.normalization.dtype).eps
        if not torch.allclose(
            expected_norm,
            self.normalization,
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError("center-inclusive quintic normalization changed")


def fixed_reference_neighborhood(
    reference: Atoms,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ReferenceNeighborhood:
    """Build the immutable reference-order 6 A graph; geometry stays detached."""
    validate_reference_semantics(reference)
    receiver, sender, image, distance = neighbor_list(
        "ijSd", reference, REFERENCE_CUTOFF_A, self_interaction=False
    )
    order = np.lexsort(
        (
            image[:, 2],
            image[:, 1],
            image[:, 0],
            sender,
            receiver,
        )
    )
    receiver_t = torch.as_tensor(receiver[order], dtype=torch.long, device=device)
    sender_t = torch.as_tensor(sender[order], dtype=torch.long, device=device)
    image_t = torch.as_tensor(image[order], dtype=torch.int64, device=device)
    distance_t = torch.as_tensor(distance[order], dtype=dtype, device=device)
    weight = quintic_reference_weight(distance_t).detach()
    normalization = torch.ones(len(reference), dtype=dtype, device=device)
    normalization.index_add_(0, receiver_t, weight)
    cell = np.asarray(reference.cell, dtype=np.float64)
    translations = []
    for first in range(-2, 3):
        for second in range(-2, 3):
            if first == 0 and second == 0:
                continue
            translations.append(np.linalg.norm(first * cell[0] + second * cell[1]))
    shortest_translation = float(min(translations))
    graph = ReferenceNeighborhood(
        receiver=receiver_t,
        sender=sender_t,
        image_integer=image_t,
        reference_distance_A=distance_t.detach(),
        weight=weight,
        normalization=normalization.detach(),
        atom_count=len(reference),
        shortest_in_plane_translation_A=shortest_translation,
    )
    graph.validate()
    return graph


@dataclass
class BackgroundFields:
    covariance_A2: torch.Tensor
    trace_A2: torch.Tensor
    cross_product_square_A4: torch.Tensor
    multipolar_gate: torch.Tensor
    amplitude_gate: torch.Tensor


def multipolar_background(
    displacement_A: torch.Tensor, graph: ReferenceNeighborhood
) -> BackgroundFields:
    """Evaluate invariant node fields from live reference-order displacements."""
    graph.validate()
    if displacement_A.shape != (graph.atom_count, 3):
        raise ValueError("displacement shape differs from reference graph")
    if displacement_A.dtype != graph.weight.dtype:
        raise ValueError("displacement/reference graph dtype differs")
    neighbor_q = graph.weight / graph.normalization[graph.receiver]
    center_q = 1.0 / graph.normalization
    weighted_neighbor_sum = displacement_A.new_zeros((graph.atom_count, 3))
    weighted_neighbor_sum.index_add_(
        0, graph.receiver, graph.weight[:, None] * displacement_A[graph.sender]
    )
    mean = (displacement_A + weighted_neighbor_sum) / graph.normalization[:, None]
    neighbor_centered = displacement_A[graph.sender] - mean[graph.receiver]
    center_centered = displacement_A - mean
    covariance = displacement_A.new_zeros((graph.atom_count, 3, 3))
    covariance.index_add_(
        0,
        graph.receiver,
        neighbor_q[:, None, None]
        * neighbor_centered[:, :, None]
        * neighbor_centered[:, None, :],
    )
    covariance = covariance + (
        center_q[:, None, None]
        * center_centered[:, :, None]
        * center_centered[:, None, :]
    )
    trace = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(-1)

    # Direct unordered-pair sum.  This is intentionally not evaluated as
    # 0.5*((tr C)^2-tr(C^2)), whose cancellation can make rank-1 b negative.
    b_nodes = []
    for atom in range(graph.atom_count):
        selected = torch.nonzero(graph.receiver == atom, as_tuple=False).flatten()
        local_vectors = torch.cat(
            (center_centered[atom : atom + 1], neighbor_centered[selected]), dim=0
        )
        local_q = torch.cat(
            (center_q[atom : atom + 1], neighbor_q[selected]), dim=0
        )
        left, right = torch.triu_indices(
            selected.numel() + 1,
            selected.numel() + 1,
            offset=1,
            device=displacement_A.device,
        )
        cross = torch.linalg.cross(local_vectors[left], local_vectors[right], dim=-1)
        b_nodes.append(
            torch.sum(local_q[left] * local_q[right] * torch.sum(cross * cross, dim=-1))
        )
    b = torch.stack(b_nodes)
    a = -torch.expm1(-b / MULTIPOLAR_BETA_A4)
    c = trace / (trace + TRACE_SCALE_A2)
    return BackgroundFields(
        covariance_A2=covariance,
        trace_A2=trace,
        cross_product_square_A4=b,
        multipolar_gate=a,
        amplitude_gate=c,
    )


@dataclass
class OrderedBackground:
    fields: BackgroundFields
    displacement_A: torch.Tensor
    assignment: Assignment
    reference: Atoms
    graph: ReferenceNeighborhood


def background_from_structure(
    structure: Atoms,
    reference_template: Atoms,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = True,
) -> OrderedBackground:
    """Order/MIC-safe structure adapter without reading any file."""
    validate_reference_semantics(reference_template)
    reference = adapt_reference_cell(reference_template, structure)
    validate_reference_semantics(reference)
    assignment = solve_assignment(structure, reference)
    ordered = reordered_structure(structure, assignment)
    positions = torch.as_tensor(
        np.asarray(ordered.positions, float), dtype=dtype, device=device
    ).clone()
    positions.requires_grad_(requires_grad)
    reference_positions = torch.as_tensor(
        np.asarray(reference.positions, float), dtype=dtype, device=device
    )
    cell = torch.as_tensor(np.asarray(reference.cell, float), dtype=dtype, device=device)
    image = torch.as_tensor(
        assignment.image_integer_reference_order, dtype=dtype, device=device
    )
    displacement = positions - image @ cell - reference_positions
    graph = fixed_reference_neighborhood(reference, device=device, dtype=dtype)
    return OrderedBackground(
        fields=multipolar_background(displacement, graph),
        displacement_A=displacement,
        assignment=assignment,
        reference=reference,
        graph=graph,
    )


@dataclass
class R2QNodeObservables:
    scale_shift_node_energy_eV: torch.Tensor
    signed_l0: torch.Tensor


def even_l0_channels(feature: torch.Tensor, irreps) -> torch.Tensor:
    """Select exactly 16 signed 0e channels and fail closed on any 0o."""
    offset = 0
    scalar_blocks = []
    for multiplicity, irrep in irreps:
        width = int(multiplicity) * int(irrep.dim)
        if int(irrep.l) == 0:
            if int(irrep.p) != 1:
                raise ValueError("R2R rejects odd-parity 0o scalar channels")
            scalar_blocks.append(feature[:, offset : offset + width])
        offset += width
    if offset != feature.shape[1]:
        raise ValueError("feature/irrep width differs")
    if sum(block.shape[1] for block in scalar_blocks) != 16:
        raise ValueError("each R2Q interaction must expose exactly 16 signed 0e channels")
    return torch.cat(scalar_blocks, dim=1)


def r2q_node_observables(
    model: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    positions: torch.Tensor,
) -> R2QNodeObservables:
    """Exact per-node ScaleShift energy plus the two 16-channel signed l=0 sets."""
    if state_dict_sha256(model) != R2Q_ENDPOINT_STATE_SHA256:
        raise ValueError("R2R requires the exact frozen R2Q endpoint state")
    validate_mace_architecture(model)
    if positions.dtype != model_dtype(model):
        raise ValueError("positions/model dtype differs")
    vectors, lengths = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=data["edge_index"],
        shifts=data["shifts"],
    )
    node_heads = data["head"][data["batch"]].to(torch.int64)
    node_indices = torch.arange(
        positions.shape[0], device=positions.device, dtype=torch.int64
    )
    node_feats = model.node_embedding(data["node_attrs"])
    edge_attrs = model.spherical_harmonics(vectors)
    edge_feats, cutoff = model.radial_embedding(
        lengths, data["node_attrs"], data["edge_index"], model.atomic_numbers
    )
    features = []
    scalars = []
    for interaction_index, (interaction, product) in enumerate(
        zip(model.interactions, model.products, strict=True)
    ):
        node_feats, skip = interaction(
            node_attrs=data["node_attrs"],
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=data["edge_index"],
            cutoff=cutoff,
            first_layer=interaction_index == 0,
        )
        node_feats = product(node_feats=node_feats, sc=skip, node_attrs=data["node_attrs"])
        features.append(node_feats)
        scalars.append(even_l0_channels(node_feats, product.linear.irreps_out))
    signed_l0 = torch.cat(scalars, dim=1)
    if signed_l0.shape != (positions.shape[0], SIGNED_L0_CHANNELS):
        raise ValueError("R2Q signed l=0 layout changed")

    node_terms = [positions.new_zeros((positions.shape[0],))]
    for readout_index, readout in enumerate(model.readouts):
        feature_index = -1 if len(model.readouts) == 1 else readout_index
        node_terms.append(
            readout(features[feature_index], node_heads)[node_indices, node_heads]
        )
    node_energy = model.scale_shift(torch.stack(node_terms).sum(0), node_heads)
    return R2QNodeObservables(node_energy, signed_l0)


def parameter_energy_basis(
    observables: R2QNodeObservables, fields: BackgroundFields
) -> torch.Tensor:
    """Return canonical raw 65-parameter basis (each column has zero 2-jet)."""
    node_energy = observables.scale_shift_node_energy_eV
    signed = observables.signed_l0
    if node_energy.ndim != 1 or signed.shape != (node_energy.numel(), 32):
        raise ValueError("R2Q node-observable shape changed")
    if fields.multipolar_gate.shape != node_energy.shape:
        raise ValueError("background/node-observable atom counts differ")
    basis = torch.cat(
        (
            (fields.multipolar_gate[:, None] * signed).sum(0),
            torch.sum(fields.multipolar_gate * fields.amplitude_gate).reshape(1),
            (
                fields.multipolar_gate[:, None]
                * fields.amplitude_gate[:, None]
                * signed
            ).sum(0),
        )
    )
    if basis.shape != (LINEAR_DESIGN_WIDTH,):
        raise AssertionError("R2R linear basis width changed")
    return basis


def _legacy_raw_a_epsilon_energy(
    observables: R2QNodeObservables, fields: BackgroundFields
) -> torch.Tensor:
    """Private negative-regression diagnostic; never a production component."""
    return torch.sum(
        fields.multipolar_gate * observables.scale_shift_node_energy_eV
    )


@dataclass
class TaylorNullLinearDesign:
    fixed_offset_energy_eV: torch.Tensor
    fixed_offset_force_eV_A: torch.Tensor
    energy_design_eV: torch.Tensor
    force_design_eV_A: torch.Tensor
    raw_affine_terms_eV: torch.Tensor
    reference_affine_terms_eV: torch.Tensor


FORMAL_GRAPH_TENSOR_KEYS = (
    "positions",
    "edge_index",
    "shifts",
    "unit_shifts",
    "cell",
    "node_attrs",
    "batch",
    "ptr",
    "head",
    "pbc",
)


def tensor_mapping_semantic_sha256(
    data: Mapping[str, torch.Tensor], keys: tuple[str, ...]
) -> str:
    digest = hashlib.sha256()
    for key in keys:
        if key not in data or not torch.is_tensor(data[key]):
            raise ValueError(f"formal graph tensor missing: {key}")
        array = data[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _verify_formal_graph_data(
    supplied: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor]
) -> str:
    for key in FORMAL_GRAPH_TENSOR_KEYS:
        if key not in supplied or not torch.is_tensor(supplied[key]):
            raise ValueError(f"supplied formal R2O graph is missing {key}")
        left = supplied[key].detach().cpu()
        right = expected[key].detach().cpu()
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(f"supplied formal R2O graph tensor changed: {key}")
    observed = tensor_mapping_semantic_sha256(supplied, FORMAL_GRAPH_TENSOR_KEYS)
    expected_hash = tensor_mapping_semantic_sha256(expected, FORMAL_GRAPH_TENSOR_KEYS)
    if observed != expected_hash:
        raise ValueError("supplied formal R2O graph semantic hash changed")
    return observed


RIGID_VECTOR_GRAPH_TENSOR_KEYS = ("positions", "shifts", "cell")
RIGID_INTERNAL_COVARIANCE_ATOL_A = 1.0e-12
RIGID_INVARIANT_GRAPH_TENSOR_KEYS = (
    "edge_index",
    "unit_shifts",
    "node_attrs",
    "batch",
    "ptr",
    "head",
    "pbc",
)


def _covariant_formal_graph(
    baseline: Mapping[str, torch.Tensor], transformation: np.ndarray
) -> dict[str, torch.Tensor]:
    """Derive an O(3) probe graph without rebuilding its neighbor-list order."""
    for key in FORMAL_GRAPH_TENSOR_KEYS:
        if key not in baseline or not torch.is_tensor(baseline[key]):
            raise ValueError(f"baseline formal graph tensor missing: {key}")
    matrix = np.asarray(transformation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rigid graph transform must be 3x3")
    result = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in baseline.items()
    }
    for key in RIGID_VECTOR_GRAPH_TENSOR_KEYS:
        value = baseline[key]
        q_transpose = torch.as_tensor(
            matrix.T, dtype=value.dtype, device=value.device
        )
        if np.array_equal(matrix, np.eye(3, dtype=np.float64)):
            result[key] = value.clone()
        else:
            result[key] = torch.matmul(value, q_transpose)
    for key in RIGID_INVARIANT_GRAPH_TENSOR_KEYS:
        if not torch.equal(result[key], baseline[key]):
            raise RuntimeError(f"rigid graph invariant tensor changed: {key}")
    return result


def rigid_transform_native_rebuild_diagnostic(
    baseline: Mapping[str, torch.Tensor],
    derived: Mapping[str, torch.Tensor],
    native_rebuilt: Mapping[str, torch.Tensor],
    transformation: np.ndarray,
) -> dict:
    """Compare native rebuild with the covariant graph; never authorize O3."""
    matrix = np.asarray(transformation, dtype=np.float64)
    base_vectors, base_lengths = get_edge_vectors_and_lengths(
        positions=baseline["positions"],
        edge_index=baseline["edge_index"],
        shifts=baseline["shifts"],
    )
    derived_vectors, derived_lengths = get_edge_vectors_and_lengths(
        positions=derived["positions"],
        edge_index=derived["edge_index"],
        shifts=derived["shifts"],
    )
    q_transpose = torch.as_tensor(
        matrix.T, dtype=base_vectors.dtype, device=base_vectors.device
    )
    tensor_differences = {}
    for key in RIGID_VECTOR_GRAPH_TENSOR_KEYS:
        tensor_differences[key] = float(
            torch.max(torch.abs(native_rebuilt[key] - derived[key])).detach()
        )
    invariant_equal = {
        key: bool(torch.equal(native_rebuilt[key], derived[key]))
        for key in RIGID_INVARIANT_GRAPH_TENSOR_KEYS
    }
    def physical_records(data: Mapping[str, torch.Tensor]) -> dict[tuple[int, ...], int]:
        edge = data["edge_index"].detach().cpu().numpy()
        unit = data["unit_shifts"].detach().cpu().numpy()
        records = {}
        for index in range(edge.shape[1]):
            key = (
                int(edge[0, index]),
                int(edge[1, index]),
                int(unit[index, 0]),
                int(unit[index, 1]),
                int(unit[index, 2]),
            )
            if key in records:
                raise ValueError("formal graph physical directed edge is duplicated")
            records[key] = index
        return records

    derived_records = physical_records(derived)
    native_records = physical_records(native_rebuilt)
    physical_keys_equal = set(derived_records) == set(native_records)
    matched_vector_max = math.inf
    matched_length_max = math.inf
    if physical_keys_equal:
        native_vectors, native_lengths = get_edge_vectors_and_lengths(
            positions=native_rebuilt["positions"],
            edge_index=native_rebuilt["edge_index"],
            shifts=native_rebuilt["shifts"],
        )
        matched_vector_max = 0.0
        matched_length_max = 0.0
        for key in sorted(derived_records):
            left = derived_records[key]
            right = native_records[key]
            matched_vector_max = max(
                matched_vector_max,
                float(
                    torch.max(
                        torch.abs(derived_vectors[left] - native_vectors[right])
                    ).detach()
                ),
            )
            matched_length_max = max(
                matched_length_max,
                abs(
                    float(derived_lengths[left].detach())
                    - float(native_lengths[right].detach())
                ),
            )
    return {
        "format": "graphene_r2r_rigid_native_rebuild_diagnostic_v1",
        "authorization_role": "diagnostic_only_not_physics_gate",
        "baseline_graph_sha256": tensor_mapping_semantic_sha256(
            baseline, FORMAL_GRAPH_TENSOR_KEYS
        ),
        "covariant_derived_graph_sha256": tensor_mapping_semantic_sha256(
            derived, FORMAL_GRAPH_TENSOR_KEYS
        ),
        "native_rebuilt_graph_sha256": tensor_mapping_semantic_sha256(
            native_rebuilt, FORMAL_GRAPH_TENSOR_KEYS
        ),
        "native_minus_covariant_vector_tensor_max_abs": tensor_differences,
        "native_invariant_tensor_byte_equal": invariant_equal,
        "native_physical_edge_identity_multiset_equal": physical_keys_equal,
        "native_matched_physical_edge_vector_max_abs_A": matched_vector_max,
        "native_matched_physical_edge_length_max_abs_A": matched_length_max,
        "native_physical_edge_multiset_equivalent": bool(physical_keys_equal),
        "numeric_differences_are_diagnostic_only": True,
        "covariant_edge_vector_max_abs_difference_A": float(
            torch.max(
                torch.abs(derived_vectors - torch.matmul(base_vectors, q_transpose))
            ).detach()
        ),
        "covariant_edge_length_max_abs_difference_A": float(
            torch.max(torch.abs(derived_lengths - base_lengths)).detach()
        ),
        "physics_gate_authorized": False,
    }


def reference_neighborhood_semantic_sha256(graph: ReferenceNeighborhood) -> str:
    """Hash the exact scalar/topological background data reused by rigid probes."""
    graph.validate()
    digest = hashlib.sha256()
    for name, value in (
        ("receiver", graph.receiver),
        ("sender", graph.sender),
        ("image_integer", graph.image_integer),
        ("reference_distance_A", graph.reference_distance_A),
        ("weight", graph.weight),
        ("normalization", graph.normalization),
    ):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes())
        digest.update(array.tobytes(order="C"))
    digest.update(np.asarray([graph.atom_count], dtype=np.dtype("<i8")).tobytes())
    digest.update(
        np.asarray(
            [graph.shortest_in_plane_translation_A], dtype=np.dtype("<f8")
        ).tobytes()
    )
    return digest.hexdigest()


def reference_neighborhood_physical_multiset_receipt(
    baseline: ReferenceNeighborhood, native_rebuilt: ReferenceNeighborhood
) -> dict:
    """Compare rotation-invariant receiver/sender/image edge identities."""
    baseline.validate()
    native_rebuilt.validate()

    def records(graph: ReferenceNeighborhood) -> dict[tuple[int, ...], int]:
        receiver = graph.receiver.detach().cpu().numpy()
        sender = graph.sender.detach().cpu().numpy()
        image = graph.image_integer.detach().cpu().numpy()
        result = {}
        for index in range(receiver.size):
            key = (
                int(receiver[index]),
                int(sender[index]),
                int(image[index, 0]),
                int(image[index, 1]),
                int(image[index, 2]),
            )
            if key in result:
                raise ValueError("reference neighborhood edge identity is duplicated")
            result[key] = index
        return result

    baseline_records = records(baseline)
    native_records = records(native_rebuilt)
    keys_equal = set(baseline_records) == set(native_records)
    distance_max = math.inf
    weight_max = math.inf
    if keys_equal:
        distance_max = 0.0
        weight_max = 0.0
        for key in sorted(baseline_records):
            left = baseline_records[key]
            right = native_records[key]
            distance_max = max(
                distance_max,
                abs(
                    float(baseline.reference_distance_A[left].detach())
                    - float(native_rebuilt.reference_distance_A[right].detach())
                ),
            )
            weight_max = max(
                weight_max,
                abs(
                    float(baseline.weight[left].detach())
                    - float(native_rebuilt.weight[right].detach())
                ),
            )
    normalization_max = float(
        torch.max(
            torch.abs(baseline.normalization - native_rebuilt.normalization)
        ).detach()
    )
    shortest_translation_abs = abs(
        baseline.shortest_in_plane_translation_A
        - native_rebuilt.shortest_in_plane_translation_A
    )
    return {
        "baseline_edge_count": len(baseline_records),
        "native_rebuilt_edge_count": len(native_records),
        "edge_identity_multiset_equal": keys_equal,
        "stored_order_byte_equal": bool(
            torch.equal(baseline.receiver, native_rebuilt.receiver)
            and torch.equal(baseline.sender, native_rebuilt.sender)
            and torch.equal(baseline.image_integer, native_rebuilt.image_integer)
        ),
        "matched_reference_distance_max_abs_A": distance_max,
        "matched_quintic_weight_max_abs": weight_max,
        "normalization_max_abs": normalization_max,
        "shortest_in_plane_translation_abs_difference_A": (
            shortest_translation_abs
        ),
        "physical_multiset_equivalent": bool(keys_equal),
        "numeric_differences_are_diagnostic_only": True,
    }


def _array_semantic_sha256(items: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def full_structure_semantic_sha256(structure: Atoms) -> str:
    """Hash numbers, pbc, cell and absolute positions without label fields."""
    return _array_semantic_sha256(
        (
            ("numbers", np.asarray(structure.numbers, dtype=np.dtype("<i8"))),
            ("pbc", np.asarray(structure.pbc, dtype=np.bool_)),
            ("cell_A", np.asarray(structure.cell, dtype=np.dtype("<f8"))),
            (
                "positions_A",
                np.asarray(structure.positions, dtype=np.dtype("<f8")),
            ),
        )
    )


@dataclass
class ProductionLinearDesignQuery:
    fixed_offset_energy_eV: torch.Tensor
    fixed_offset_force_source_order_eV_A: torch.Tensor
    parameter_energy_design_eV: torch.Tensor
    parameter_force_design_source_order_eV_A: torch.Tensor
    assignment: Assignment
    receipt: dict
    _live_current_positions_reference_order: torch.Tensor

    def canonical_combined_probe(self) -> "ProductionCombinedProbe":
        """Combine the fixed carrier and frozen mechanics probe inside the API.

        Formal mechanics callers must use this method instead of assembling
        ``fixed + X @ p`` independently.  The coefficient semantic hash is
        therefore checked at the same production boundary as the graph,
        endpoint, reference, assignment and MIC provenance.
        """
        coefficient_hash = semantic_sha256(list(MECHANICS_PROBE_COEFFICIENTS))
        if coefficient_hash != MECHANICS_PROBE_COEFFICIENTS_SHA256:
            raise RuntimeError("canonical mechanics coefficient constant changed")
        coefficients = torch.as_tensor(
            MECHANICS_PROBE_COEFFICIENTS,
            dtype=self.parameter_energy_design_eV.dtype,
            device=self.parameter_energy_design_eV.device,
        )
        energy = self.fixed_offset_energy_eV[0] + torch.dot(
            self.parameter_energy_design_eV, coefficients
        )
        force = self.fixed_offset_force_source_order_eV_A[:, 0] + (
            self.parameter_force_design_source_order_eV_A @ coefficients
        )
        atom_count = force.numel() // 3
        return ProductionCombinedProbe(
            energy_eV=energy,
            force_source_order_eV_A=force.reshape(atom_count, 3),
            coefficients_sha256=coefficient_hash,
            query_receipt=self.receipt,
        )


@dataclass
class ProductionCombinedProbe:
    energy_eV: torch.Tensor
    force_source_order_eV_A: torch.Tensor
    coefficients_sha256: str
    query_receipt: dict


@dataclass
class _VerifiedProductionContext:
    node_observables_fn: Callable[[torch.Tensor], R2QNodeObservables]
    background_fields_fn: Callable[[torch.Tensor], BackgroundFields]
    parameter_basis_fn: Callable[[R2QNodeObservables, BackgroundFields], torch.Tensor]
    current_positions_reference_order: torch.Tensor
    reference_positions: torch.Tensor
    image_shift: torch.Tensor
    assignment: Assignment
    atom_count: int
    receipt: dict


def native_cell_wrap_order_mic_receipt(
    baseline: ProductionLinearDesignQuery,
    native_wrapped: ProductionLinearDesignQuery,
) -> dict:
    """Evaluate the native-cell wrap gate on the canonical combined probe.

    Raw 65-column sensitivity is retained in the receipt for diagnosis, but it
    is deliberately excluded from the pass decision because the archived ASE
    cell and frozen float32-origin graph cell differ at their last bits.
    """
    provenance_keys = (
        "endpoint_state_sha256",
        "reference_semantic_sha256",
        "formal_R2O_graph_semantic_sha256",
        "formal_R2O_graph_mode",
    )
    for key in provenance_keys:
        if baseline.receipt[key] != native_wrapped.receipt[key]:
            raise ValueError(f"native-cell wrap comparison changed {key}")
    base_combined = baseline.canonical_combined_probe()
    wrapped_combined = native_wrapped.canonical_combined_probe()
    energy_abs = abs(
        float((wrapped_combined.energy_eV - base_combined.energy_eV).detach())
    )
    force_abs = float(
        torch.max(
            torch.abs(
                wrapped_combined.force_source_order_eV_A
                - base_combined.force_source_order_eV_A
            )
        ).detach()
    )
    raw_difference = torch.abs(
        native_wrapped.parameter_force_design_source_order_eV_A
        - baseline.parameter_force_design_source_order_eV_A
    )
    raw_max_abs = float(torch.max(raw_difference).detach())
    raw_scale = float(
        torch.max(torch.abs(baseline.parameter_force_design_source_order_eV_A)).detach()
    )
    raw_relative = raw_max_abs / max(raw_scale, torch.finfo(torch.float64).tiny)
    energy_limit = CANONICAL_CONTRACT["fixed_gates"][
        "native_cell_wrap_order_MIC_combined_probe"
    ]["energy_abs_eV"]
    force_limit = CANONICAL_CONTRACT["fixed_gates"][
        "native_cell_wrap_order_MIC_combined_probe"
    ]["force_max_abs_eV_A"]
    return {
        "format": "graphene_r2r_native_cell_wrap_order_mic_receipt_v1",
        "authorization_observable": "canonical fixed_offset + X @ p",
        "coefficients_sha256": MECHANICS_PROBE_COEFFICIENTS_SHA256,
        "combined_energy_abs_eV": energy_abs,
        "combined_force_max_abs_eV_A": force_abs,
        "combined_energy_limit_eV": energy_limit,
        "combined_force_limit_eV_A": force_limit,
        "raw_65_force_design_max_abs_eV_A": raw_max_abs,
        "raw_65_force_design_relative": raw_relative,
        "raw_65_column_difference_role": "diagnostic_only_not_authorization",
        "pass": bool(energy_abs <= energy_limit and force_abs <= force_limit),
    }


def taylor_null_linear_force_design(
    basis_fn: Callable[[torch.Tensor], torch.Tensor],
    current_positions: torch.Tensor,
    reference_positions: torch.Tensor,
    image_shift: torch.Tensor,
    *,
    create_graph: bool = False,
) -> TaylorNullLinearDesign:
    """Apply a separate whole-energy Taylor-2 null to every linear column."""
    if current_positions.shape != reference_positions.shape:
        raise ValueError("current/reference positions differ")
    if not current_positions.requires_grad:
        raise ValueError("current positions must be live")
    if image_shift.requires_grad:
        raise ValueError("MIC image shift must remain detached")
    x = current_positions
    x0 = reference_positions.detach().clone().requires_grad_(True)
    aligned = x - image_shift.detach()
    displacement = aligned - x0.detach()
    raw = basis_fn(aligned)
    reference = basis_fn(x0)
    expected_width = LINEAR_DESIGN_WIDTH + 1
    if raw.shape != (expected_width,) or reference.shape != raw.shape:
        raise ValueError("affine basis must contain fixed offset plus 65 columns")
    remainder_columns = []
    for column in range(expected_width):
        gradient = torch.autograd.grad(
            reference[column], x0, create_graph=True, retain_graph=True
        )[0]
        hvp = torch.autograd.grad(
            gradient,
            x0,
            grad_outputs=displacement,
            create_graph=True,
            retain_graph=True,
        )[0]
        remainder_columns.append(
            raw[column]
            - reference[column]
            - torch.sum(gradient * displacement)
            - 0.5 * torch.sum(displacement * hvp)
        )
    remainder = torch.stack(remainder_columns)
    force_columns = []
    for column in range(expected_width):
        force = -torch.autograd.grad(
            remainder[column],
            x,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        force_columns.append(force.reshape(-1))
    force = torch.stack(force_columns, dim=1)
    if not create_graph:
        remainder = remainder.detach()
        force = force.detach()
        raw = raw.detach()
        reference = reference.detach()
    return TaylorNullLinearDesign(
        fixed_offset_energy_eV=remainder[:1],
        fixed_offset_force_eV_A=force[:, :1],
        energy_design_eV=remainder[1:],
        force_design_eV_A=force[:, 1:],
        raw_affine_terms_eV=raw,
        reference_affine_terms_eV=reference,
    )


def _local_direct_multipolar_b(
    local_displacement_A: torch.Tensor, normalized_q: torch.Tensor
) -> torch.Tensor:
    """Evaluate one receiver's direct unordered-pair ``b``."""
    if local_displacement_A.shape != (normalized_q.numel(), 3):
        raise ValueError("local displacement/q shape changed")
    mean = torch.sum(normalized_q[:, None] * local_displacement_A, dim=0)
    centered = local_displacement_A - mean
    left, right = torch.triu_indices(
        normalized_q.numel(), normalized_q.numel(), offset=1,
        device=local_displacement_A.device,
    )
    cross = torch.linalg.cross(centered[left], centered[right], dim=-1)
    return torch.sum(
        normalized_q[left]
        * normalized_q[right]
        * torch.sum(cross * cross, dim=-1)
    )


def _local_direct_multipolar_gate(
    local_displacement_A: torch.Tensor, normalized_q: torch.Tensor
) -> torch.Tensor:
    """Evaluate one receiver's gate on center+neighbor samples."""
    b = _local_direct_multipolar_b(local_displacement_A, normalized_q)
    return -torch.expm1(-b / MULTIPOLAR_BETA_A4)


def _local_q_signature(
    center_q: torch.Tensor, neighbor_q: torch.Tensor, sender: torch.Tensor
) -> tuple[str, dict]:
    """Hash exact FP64 q bytes and sender multiplicity without tolerance grouping."""
    center = np.asarray([float(center_q.detach())], dtype=np.dtype("<f8"))
    sorted_q = np.sort(
        neighbor_q.detach().cpu().numpy().astype(np.dtype("<f8"), copy=False)
    )
    _, counts = torch.unique(sender.detach().cpu(), return_counts=True)
    counts_np = np.sort(counts.numpy().astype(np.dtype("<i8"), copy=False))
    sender_unique = bool(torch.all(counts == 1))
    digest = hashlib.sha256()
    digest.update(center.tobytes())
    digest.update(sorted_q.tobytes())
    digest.update(counts_np.tobytes())
    digest.update(bytes([int(sender_unique)]))
    return digest.hexdigest(), {
        "center_q_hex": float(center[0]).hex(),
        "neighbor_count": int(neighbor_q.numel()),
        "sender_unique": sender_unique,
        "sender_multiplicity_sorted": counts_np.tolist(),
    }


def background_rank0_zero_jet_audit(
    graph: ReferenceNeighborhood, *, full_local_hessian: bool = True
) -> dict:
    """Fail-closed zero-jet audit using every node's independent local graph.

    The mathematical authorization is the direct-pair ``a_i=O(u^4)`` proof.
    Numerically, all global node values/Jacobians are checked.  A fresh local
    center+neighbor graph then checks the full local Hessian for every node;
    exact q signatures are recorded for coverage, never tolerance-clustered.
    """
    started = time.perf_counter()
    graph.validate()
    if graph.weight.dtype != torch.float64:
        raise ValueError("zero-jet audit requires exact FP64 normalized-q semantics")
    displacement = torch.zeros(
        (graph.atom_count, 3),
        dtype=graph.weight.dtype,
        device=graph.weight.device,
        requires_grad=True,
    )
    gate = multipolar_background(displacement, graph).multipolar_gate
    global_gradient_max = []
    for node in range(graph.atom_count):
        gradient = torch.autograd.grad(
            gate[node], displacement, retain_graph=True
        )[0]
        global_gradient_max.append(float(torch.max(torch.abs(gradient)).detach()))

    neighbor_q_all = graph.weight / graph.normalization[graph.receiver]
    center_q_all = 1.0 / graph.normalization
    orbit_nodes: dict[str, list[int]] = {}
    signature_metadata: dict[str, dict] = {}
    selected_by_node: list[torch.Tensor] = []
    local_q_by_node: list[torch.Tensor] = []
    local_source_order_payload = []
    for node in range(graph.atom_count):
        selected = torch.nonzero(graph.receiver == node, as_tuple=False).flatten()
        sender = graph.sender[selected]
        signature, metadata = _local_q_signature(
            center_q_all[node], neighbor_q_all[selected], sender
        )
        orbit_nodes.setdefault(signature, []).append(node)
        signature_metadata.setdefault(signature, metadata)
        selected_by_node.append(selected)
        local_source_order_payload.append(
            {"receiver": node, "sender_order": sender.detach().cpu().tolist()}
        )
        local_q_by_node.append(
            torch.cat((center_q_all[node : node + 1], neighbor_q_all[selected]))
        )
    covered = [node for nodes in orbit_nodes.values() for node in nodes]
    coverage_exact_once = sorted(covered) == list(range(graph.atom_count)) and (
        len(covered) == len(set(covered))
    )
    all_local_senders_unique = all(
        metadata["sender_unique"] for metadata in signature_metadata.values()
    )
    orbit_payload = [
        {"signature": signature, "nodes": nodes, **signature_metadata[signature]}
        for signature, nodes in sorted(orbit_nodes.items())
    ]
    receipt = {
        "format": "graphene_r2r_background_rank0_zero_jet_audit_v3_local_direct",
        "proof": "direct unordered cross-product-square gives b_i=O(u^4), hence a_i=O(u^4)",
        "multipolar_gate_value_max_abs": float(torch.max(torch.abs(gate)).detach()),
        "global_all_node_Jacobian_max_abs_A-1": max(global_gradient_max),
        "global_all_node_value_and_Jacobian_exact_zero": bool(
            torch.count_nonzero(gate.detach()) == 0
            and all(value == 0.0 for value in global_gradient_max)
        ),
        "q_signature_definition": (
            "exact little-endian FP64 bytes of center q and sorted neighbor-q "
            "multiset plus exact sender multiplicities/unique flag; no rounding"
        ),
        "q_signature_orbits": orbit_payload,
        "q_signature_orbit_count": len(orbit_payload),
        "q_signature_orbits_sha256": semantic_sha256(orbit_payload),
        "all_nodes_covered_exactly_once": coverage_exact_once,
        "production_local_source_order_sha256": semantic_sha256(
            local_source_order_payload
        ),
        "all_local_senders_unique": all_local_senders_unique,
        "full_local_hessian_performed": bool(full_local_hessian),
    }
    if not full_local_hessian:
        return {**receipt, "pass": False}

    local_value_max = []
    local_gradient_max = []
    local_hessian_max = []
    for local_q in local_q_by_node:
        local_zero = torch.zeros(
            (local_q.numel(), 3), dtype=local_q.dtype, device=local_q.device,
            requires_grad=True,
        )
        local_value = _local_direct_multipolar_gate(local_zero, local_q)
        local_gradient = torch.autograd.grad(
            local_value, local_zero, create_graph=True
        )[0]
        local_hessian = torch.autograd.functional.hessian(
            lambda z: _local_direct_multipolar_gate(z, local_q),
            local_zero.detach(),
            create_graph=False,
            vectorize=True,
        )
        local_value_max.append(abs(float(local_value.detach())))
        local_gradient_max.append(float(torch.max(torch.abs(local_gradient)).detach()))
        local_hessian_max.append(float(torch.max(torch.abs(local_hessian)).detach()))
        del local_value, local_gradient, local_hessian, local_zero

    # Four evenly spaced receivers bind the independent local evaluator back
    # to the production global implementation away from the zero jet.  The
    # signed base is scaled per receiver so b/beta=1 and a is not saturated.
    probe_nodes = sorted(
        set(np.linspace(0, graph.atom_count - 1, 4, dtype=int).tolist())
    )
    sample_count = int(local_q_by_node[0].numel())
    base = torch.arange(
        sample_count * 3, dtype=graph.weight.dtype, device=graph.weight.device
    )
    base = (
        torch.sin(0.37 * base + 0.19) + torch.cos(0.23 * base - 0.11)
    ).reshape(sample_count, 3)
    base = 1.0e-3 * (base - base.mean(0))
    probe_receipts = []
    for node in probe_nodes:
        selected = selected_by_node[node]
        sender = graph.sender[selected]
        local_q = local_q_by_node[node]
        base_b = _local_direct_multipolar_b(base, local_q)
        scale = (MULTIPOLAR_BETA_A4 / base_b).pow(0.25).detach()
        local_z = (scale * base).detach().requires_grad_(True)
        local_value = _local_direct_multipolar_gate(local_z, local_q)
        local_gradient = torch.autograd.grad(local_value, local_z)[0]
        global_z = torch.zeros(
            (graph.atom_count, 3), dtype=graph.weight.dtype, device=graph.weight.device
        )
        global_z[node] = local_z.detach()[0]
        global_z.index_copy_(0, sender, local_z.detach()[1:])
        global_z.requires_grad_(True)
        production_value = multipolar_background(global_z, graph).multipolar_gate[node]
        production_gradient = torch.autograd.grad(production_value, global_z)[0]
        mapped_gradient = torch.zeros_like(production_gradient)
        mapped_gradient[node] = local_gradient[0]
        mapped_gradient.index_add_(0, sender, local_gradient[1:])
        touched = torch.zeros(graph.atom_count, dtype=torch.bool, device=graph.weight.device)
        touched[node] = True
        touched[sender] = True
        outside_gradient = production_gradient[~touched]
        outside_gradient_max = (
            float(torch.max(torch.abs(outside_gradient)).detach())
            if outside_gradient.numel()
            else 0.0
        )
        probe_receipts.append(
            {
                "node": node,
                "base_b_A4": float(base_b),
                "scale": float(scale),
                "b_over_beta": float(base_b * scale**4 / MULTIPOLAR_BETA_A4),
                "target_a": float(-math.expm1(-1.0)),
                "local_a": float(local_value.detach()),
                "production_a": float(production_value.detach()),
                "value_abs_difference": abs(float((production_value-local_value).detach())),
                "mapped_gradient_max_abs_difference_A-1": float(
                    torch.max(torch.abs(production_gradient-mapped_gradient)).detach()
                ),
                "outside_local_global_gradient_max_abs_A-1": float(
                    outside_gradient_max
                ),
            }
        )
    finite = all(
        math.isfinite(value)
        for value in (*local_value_max, *local_gradient_max, *local_hessian_max)
    )
    local_all_zero = all(value == 0.0 for value in local_value_max) and all(
        value == 0.0 for value in local_gradient_max
    ) and all(value == 0.0 for value in local_hessian_max)
    parity_ok = all(
        all(math.isfinite(float(value)) for value in item.values())
        and item["base_b_A4"] > 0.0
        and item["scale"] > 0.0
        and abs(item["b_over_beta"] - 1.0) <= 1.0e-12
        and abs(item["local_a"] - item["target_a"]) <= 1.0e-14
        and item["value_abs_difference"] <= 1.0e-14
        and item["mapped_gradient_max_abs_difference_A-1"] <= 1.0e-12
        and item["outside_local_global_gradient_max_abs_A-1"] == 0.0
        for item in probe_receipts
    )
    return {
        **receipt,
        "node_count": graph.atom_count,
        "local_sample_count": int(local_q_by_node[0].numel()),
        "local_Hessian_shape_per_node": [sample_count, 3, sample_count, 3],
        "local_all_node_value_max_abs": max(local_value_max),
        "local_all_node_Jacobian_max_abs_A-1": max(local_gradient_max),
        "local_all_node_Hessian_max_abs_A-2": max(local_hessian_max),
        "local_all_node_finite": finite,
        "nonzero_local_production_probe": probe_receipts,
        "nonzero_probe_base_sha256": hashlib.sha256(
            base.detach().cpu().numpy().astype(np.dtype("<f8"), copy=False).tobytes()
        ).hexdigest(),
        "nonzero_probe_nodes_sha256": semantic_sha256(probe_nodes),
        "elapsed_seconds": time.perf_counter() - started,
        "process_ru_maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "pass": bool(
            receipt["global_all_node_value_and_Jacobian_exact_zero"]
            and coverage_exact_once
            and all_local_senders_unique
            and finite
            and local_all_zero
            and parity_ok
        ),
    }


def _verified_production_context(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    *,
    device: torch.device | str,
    formal_graph_data: Mapping[str, torch.Tensor] | None = None,
    graph_mode: str = "baseline",
    baseline_reference_template: Atoms | None = None,
    baseline_structure_template: Atoms | None = None,
    rigid_transform: np.ndarray | None = None,
    _diagnostic_native_rebuild: bool = False,
) -> _VerifiedProductionContext:
    """Build the shared hash-bound production context and affine basis.

    Assignment, MIC images, the formal R2O graph, the endpoint state and the
    ordered reference are all recomputed and checked inside this boundary.
    Formal mechanics/design materializers must not assemble the primitives by
    hand.
    """
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("R2R production graph requires float32-origin default dtype")
    if graph_mode not in ("baseline", "rigid_transform_probe"):
        raise ValueError("R2R production graph mode changed")
    if _diagnostic_native_rebuild and graph_mode != "rigid_transform_probe":
        raise ValueError("native rebuild diagnostic requires rigid-transform mode")
    if state_dict_sha256(model) != R2Q_ENDPOINT_STATE_SHA256:
        raise ValueError("R2R production query requires frozen endpoint state")
    validate_mace_architecture(model)
    validate_reference_semantics(reference_template)
    reference = adapt_reference_cell(reference_template, structure)
    reference_hash = validate_reference_semantics(reference)
    assignment = solve_assignment(structure, reference)
    assignment.validate(len(reference))
    ordered = reordered_structure(structure, assignment)
    aligned_audit = ordered.copy()
    aligned_audit.positions = (
        np.asarray(ordered.positions, dtype=np.float64)
        - assignment.image_integer_reference_order @ np.asarray(ordered.cell, dtype=np.float64)
    )
    validate_current_edge_set(aligned_audit, reference)

    role = {72: "reference_6x6", 128: "reference_8x8"}[len(reference)]
    dtype = model_dtype(model)
    transform_receipt = None
    if graph_mode == "baseline":
        expected_data = fixed_reference_graph(reference, device=device, dtype=dtype)
        if formal_graph_data is None:
            data = expected_data
            graph_hash = tensor_mapping_semantic_sha256(
                data, FORMAL_GRAPH_TENSOR_KEYS
            )
        else:
            graph_hash = _verify_formal_graph_data(formal_graph_data, expected_data)
            data = formal_graph_data
        if (
            baseline_reference_template is not None
            or baseline_structure_template is not None
            or rigid_transform is not None
        ):
            raise ValueError("baseline graph mode forbids rigid-transform inputs")
        if graph_hash != EXPECTED_BASELINE_FORMAL_GRAPH_SHA256[role]:
            raise ValueError("baseline formal R2O graph hash differs from frozen hash")
        baseline_graph_hash = graph_hash
        background_graph = fixed_reference_neighborhood(
            reference, device=device, dtype=dtype
        )
    else:
        if (
            baseline_reference_template is None
            or baseline_structure_template is None
            or rigid_transform is None
        ):
            raise ValueError(
                "rigid-transform graph mode requires baseline reference, structure and Q"
            )
        validate_reference_semantics(baseline_reference_template)
        transformation = np.asarray(rigid_transform, dtype=np.float64)
        if transformation.shape != (3, 3):
            raise ValueError("rigid transform must be 3x3")
        if not np.allclose(
            transformation @ transformation.T, np.eye(3), atol=1.0e-12, rtol=0.0
        ) or not math.isclose(abs(np.linalg.det(transformation)), 1.0, abs_tol=1.0e-12):
            raise ValueError("rigid transform must be proper/improper orthogonal")
        expected_cell = np.asarray(baseline_reference_template.cell) @ transformation.T
        expected_positions = (
            np.asarray(baseline_reference_template.positions) @ transformation.T
        )
        if not np.array_equal(
            np.asarray(reference_template.cell), expected_cell
        ) or not np.array_equal(
            np.asarray(reference_template.positions), expected_positions
        ):
            raise ValueError("rigid-transform reference is not covariant with baseline")
        canonical_baseline_reference = adapt_reference_cell(
            baseline_reference_template, baseline_reference_template
        )
        validate_reference_semantics(canonical_baseline_reference)
        baseline_data = fixed_reference_graph(
            canonical_baseline_reference, device=device, dtype=dtype
        )
        baseline_graph_hash = tensor_mapping_semantic_sha256(
            baseline_data, FORMAL_GRAPH_TENSOR_KEYS
        )
        if baseline_graph_hash != EXPECTED_BASELINE_FORMAL_GRAPH_SHA256[role]:
            raise ValueError("rigid probe baseline graph differs from frozen hash")
        derived_data = _covariant_formal_graph(baseline_data, transformation)
        derived_graph_hash = tensor_mapping_semantic_sha256(
            derived_data, FORMAL_GRAPH_TENSOR_KEYS
        )
        native_rebuilt_data = fixed_reference_graph(
            reference, device=device, dtype=dtype
        )
        native_diagnostic = rigid_transform_native_rebuild_diagnostic(
            baseline_data, derived_data, native_rebuilt_data, transformation
        )
        if not native_diagnostic["native_physical_edge_multiset_equivalent"]:
            raise ValueError("native rebuilt formal graph physical edge multiset changed")
        if _diagnostic_native_rebuild:
            if formal_graph_data is not None:
                raise ValueError("native rebuild diagnostic forbids supplied graph data")
            data = native_rebuilt_data
            graph_hash = native_diagnostic["native_rebuilt_graph_sha256"]
        elif formal_graph_data is None:
            data = derived_data
            graph_hash = derived_graph_hash
        else:
            graph_hash = _verify_formal_graph_data(formal_graph_data, derived_data)
            data = formal_graph_data

        baseline_structure = baseline_structure_template.copy()
        expected_live_positions = (
            np.asarray(baseline_structure.positions, dtype=np.float64)
            @ transformation.T
        )
        expected_live_cell = (
            np.asarray(baseline_structure.cell, dtype=np.float64)
            @ transformation.T
        )
        if not np.array_equal(
            np.asarray(structure.numbers), np.asarray(baseline_structure.numbers)
        ) or not np.array_equal(
            np.asarray(structure.pbc), np.asarray(baseline_structure.pbc)
        ):
            raise ValueError("rigid live structure identity changed")
        if not np.array_equal(
            np.asarray(structure.positions), expected_live_positions
        ) or not np.array_equal(
            np.asarray(structure.cell), expected_live_cell
        ):
            raise ValueError("rigid live structure/reference covariance changed")
        baseline_reference = adapt_reference_cell(
            baseline_reference_template, baseline_structure
        )
        baseline_assignment = solve_assignment(baseline_structure, baseline_reference)
        baseline_assignment.validate(len(baseline_reference))
        assignment_arrays_equal = {
            "reference_to_source": bool(
                np.array_equal(
                    assignment.reference_to_source,
                    baseline_assignment.reference_to_source,
                )
            ),
            "source_to_reference": bool(
                np.array_equal(
                    assignment.source_to_reference,
                    baseline_assignment.source_to_reference,
                )
            ),
            "image_integer_reference_order": bool(
                np.array_equal(
                    assignment.image_integer_reference_order,
                    baseline_assignment.image_integer_reference_order,
                )
            ),
        }
        if not all(assignment_arrays_equal.values()):
            raise ValueError("rigid transform changed assignment or MIC image gauge")
        baseline_assignment_hash = _array_semantic_sha256(
            (
                (
                    "reference_to_source",
                    baseline_assignment.reference_to_source.astype(np.int64),
                ),
                (
                    "source_to_reference",
                    baseline_assignment.source_to_reference.astype(np.int64),
                ),
                (
                    "image_integer_reference_order",
                    baseline_assignment.image_integer_reference_order.astype(np.int64),
                ),
            )
        )
        baseline_ordered = reordered_structure(
            baseline_structure, baseline_assignment
        )
        live_position_covariance_max = float(
            np.max(
                np.abs(
                    np.asarray(ordered.positions, dtype=np.float64)
                    - np.asarray(baseline_ordered.positions, dtype=np.float64)
                    @ transformation.T
                )
            )
        )
        live_cell_covariance_max = float(
            np.max(
                np.abs(
                    np.asarray(ordered.cell, dtype=np.float64)
                    - np.asarray(baseline_ordered.cell, dtype=np.float64)
                    @ transformation.T
                )
            )
        )
        reference_covariance_max = float(
            np.max(
                np.abs(
                    np.asarray(reference.positions, dtype=np.float64)
                    - np.asarray(baseline_reference.positions, dtype=np.float64)
                    @ transformation.T
                )
            )
        )
        internal_covariance_within_named_tolerance = {
            "ordered_positions": bool(
                live_position_covariance_max <= RIGID_INTERNAL_COVARIANCE_ATOL_A
            ),
            "ordered_cell": bool(
                live_cell_covariance_max <= RIGID_INTERNAL_COVARIANCE_ATOL_A
            ),
            "adapted_reference_positions": bool(
                reference_covariance_max <= RIGID_INTERNAL_COVARIANCE_ATOL_A
            ),
        }
        if not all(internal_covariance_within_named_tolerance.values()):
            raise ValueError("rigid live structure/reference covariance changed")

        background_graph = fixed_reference_neighborhood(
            canonical_baseline_reference, device=device, dtype=dtype
        )
        native_background_graph = fixed_reference_neighborhood(
            reference, device=device, dtype=dtype
        )
        background_native_diagnostic = (
            reference_neighborhood_physical_multiset_receipt(
                background_graph, native_background_graph
            )
        )
        if not background_native_diagnostic["physical_multiset_equivalent"]:
            raise ValueError("rotated background physical edge multiset changed")
        transform_receipt = {
            "format": "graphene_r2r_covariant_rigid_graph_receipt_v1",
            "matrix": transformation.tolist(),
            "matrix_semantic_sha256": semantic_sha256(transformation.tolist()),
            "determinant": float(np.linalg.det(transformation)),
            "algorithm": (
                "exact frozen baseline graph; positions/shifts/cell @ Q.T; "
                "all other FORMAL_GRAPH_TENSOR_KEYS byte exact"
            ),
            "vector_tensor_keys": list(RIGID_VECTOR_GRAPH_TENSOR_KEYS),
            "invariant_tensor_keys": list(RIGID_INVARIANT_GRAPH_TENSOR_KEYS),
            "derived_graph_sha256": derived_graph_hash,
            "selected_graph_sha256": graph_hash,
            "baseline_graph_sha256": baseline_graph_hash,
            "baseline_graph_frozen_hash_match": True,
            "baseline_structure_full_semantic_sha256": (
                full_structure_semantic_sha256(baseline_structure)
            ),
            "baseline_reference_full_semantic_sha256": (
                full_structure_semantic_sha256(baseline_reference_template)
            ),
            "baseline_assignment_and_MIC_semantic_sha256": (
                baseline_assignment_hash
            ),
            "assignment_arrays_byte_equal": assignment_arrays_equal,
            "live_position_covariance_max_abs_A": live_position_covariance_max,
            "live_cell_covariance_max_abs_A": live_cell_covariance_max,
            "reference_position_covariance_max_abs_A": reference_covariance_max,
            "public_reference_input_covariance_array_exact": True,
            "public_structure_input_covariance_array_exact": True,
            "internal_covariance_atol_A": RIGID_INTERNAL_COVARIANCE_ATOL_A,
            "reordered_and_reference_covariance_within_named_tolerance": (
                internal_covariance_within_named_tolerance
            ),
            "background_graph_sha256": reference_neighborhood_semantic_sha256(
                background_graph
            ),
            "background_topology_distance_weight_normalization_source": (
                "exact frozen baseline ReferenceNeighborhood"
            ),
            "background_live_centered_q_vector_policy": (
                "synchronized live displacement/reference vectors transform by Q"
            ),
            "native_background_topology_equal": background_native_diagnostic[
                "edge_identity_multiset_equal"
            ],
            "native_background_distance_max_abs_A": background_native_diagnostic[
                "matched_reference_distance_max_abs_A"
            ],
            "native_background_physical_edge_multiset_equivalent": (
                background_native_diagnostic["physical_multiset_equivalent"]
            ),
            "native_background_multiset_diagnostic": background_native_diagnostic,
            "native_rebuild_diagnostic": native_diagnostic,
            "native_rebuild_used_for_physics": _diagnostic_native_rebuild,
        }

    positions = torch.as_tensor(
        np.asarray(ordered.positions, dtype=np.float64), dtype=dtype, device=device
    ).clone().requires_grad_(True)
    reference_positions = torch.as_tensor(
        np.asarray(reference.positions, dtype=np.float64), dtype=dtype, device=device
    )
    cell = data["cell"].reshape(-1, 3, 3)[0]
    image_integer = torch.as_tensor(
        assignment.image_integer_reference_order, dtype=torch.int64, device=device
    )
    image_shift = image_integer.to(dtype=dtype) @ cell

    def node_observables_fn(aligned_positions: torch.Tensor) -> R2QNodeObservables:
        return r2q_node_observables(model, data, aligned_positions)

    def background_fields_fn(aligned_positions: torch.Tensor) -> BackgroundFields:
        displacement = aligned_positions - reference_positions.detach()
        return multipolar_background(displacement, background_graph)

    def parameter_basis_fn(
        observables: R2QNodeObservables, fields: BackgroundFields
    ) -> torch.Tensor:
        return parameter_energy_basis(observables, fields)

    atom_count = len(reference)
    structure_hash = _array_semantic_sha256(
        (
            ("numbers", np.asarray(structure.numbers, dtype=np.int64)),
            ("pbc", np.asarray(structure.pbc, dtype=np.bool_)),
            ("cell", np.asarray(structure.cell, dtype=np.float64)),
            ("absolute_positions_source_order", np.asarray(structure.positions, dtype=np.float64)),
        )
    )
    assignment_hash = _array_semantic_sha256(
        (
            ("reference_to_source", assignment.reference_to_source.astype(np.int64)),
            ("source_to_reference", assignment.source_to_reference.astype(np.int64)),
            (
                "image_integer_reference_order",
                assignment.image_integer_reference_order.astype(np.int64),
            ),
        )
    )
    live_reference_order_hash = _array_semantic_sha256(
        (
            (
                "absolute_positions_reference_order",
                positions.detach().cpu().numpy().astype(np.float64, copy=False),
            ),
            (
                "image_integer_reference_order",
                assignment.image_integer_reference_order.astype(np.int64),
            ),
            (
                "aligned_positions_reference_order",
                aligned_audit.positions.astype(np.float64, copy=False),
            ),
        )
    )
    receipt = {
        "format": "graphene_r2r_production_linear_design_query_v2",
        "endpoint_state_sha256": R2Q_ENDPOINT_STATE_SHA256,
        "reference_semantic_sha256": reference_hash,
        "formal_R2O_graph_semantic_sha256": graph_hash,
        "formal_R2O_graph_mode": graph_mode,
        "baseline_formal_R2O_graph_sha256": baseline_graph_hash,
        "baseline_formal_R2O_graph_hash_match": (
            baseline_graph_hash == EXPECTED_BASELINE_FORMAL_GRAPH_SHA256[role]
        ),
        "rigid_transform_receipt": transform_receipt,
        "diagnostic_native_rebuild_selected": _diagnostic_native_rebuild,
        "graph_construction_default_dtype": str(torch.get_default_dtype()),
        "native_vs_formal_graph_cell_max_abs_A": float(
            np.max(
                np.abs(
                    np.asarray(ordered.cell, dtype=np.float64)
                    - cell.detach().cpu().numpy().astype(np.float64, copy=False)
                )
            )
        ),
        "absolute_structure_semantic_sha256": structure_hash,
        "assignment_and_MIC_semantic_sha256": assignment_hash,
        "live_absolute_aligned_reference_order_semantic_sha256": (
            live_reference_order_hash
        ),
        "assignment_maximum_distance_A": assignment.maximum_distance_A,
        "assignment_minimum_uniqueness_gap_A": assignment.minimum_uniqueness_gap_A,
        "shortest_in_plane_translation_A": (
            background_graph.shortest_in_plane_translation_A
        ),
        "interaction_diameter_A": BACKGROUND_INTERACTION_DIAMETER_A,
        "no_wrap_margin_A": (
            background_graph.shortest_in_plane_translation_A
            - BACKGROUND_INTERACTION_DIAMETER_A
        ),
        "source_order_force_rows": 3 * atom_count,
        "parameter_columns": LINEAR_DESIGN_WIDTH,
        "fixed_offset_columns": 1,
        "affine_component_names_sha256": AFFINE_COMPONENT_NAMES_SHA256,
        "production_integration_API_only": True,
    }
    return _VerifiedProductionContext(
        node_observables_fn=node_observables_fn,
        background_fields_fn=background_fields_fn,
        parameter_basis_fn=parameter_basis_fn,
        current_positions_reference_order=positions,
        reference_positions=reference_positions,
        image_shift=image_shift,
        assignment=assignment,
        atom_count=atom_count,
        receipt=receipt,
    )


def weighted_node_taylor2_carrier_energy(
    epsilon_current: torch.Tensor,
    epsilon_reference_fn: Callable[[torch.Tensor], torch.Tensor],
    live_weight: torch.Tensor,
    current_positions: torch.Tensor,
    reference_positions: torch.Tensor,
    image_shift: torch.Tensor,
) -> torch.Tensor:
    """Return ``sum_i a_i T2null[epsilon_i]`` with one weighted jet.

    ``live_weight`` must remain connected to the current geometry.  It is
    independent of the temporary reference variable, so differentiating
    ``dot(a_live, epsilon(x0))`` produces the exactly weighted node gradients
    and Hessian-vector product while preserving all ``d a`` product terms in
    the final current-position derivative.
    """
    if not current_positions.requires_grad or not live_weight.requires_grad:
        raise ValueError("carrier positions and live multipolar weights must be live")
    if image_shift.requires_grad:
        raise ValueError("MIC image shift must remain detached")
    aligned = current_positions - image_shift.detach()
    displacement = aligned - reference_positions.detach()
    if epsilon_current.shape != live_weight.shape:
        raise ValueError("node energy/live weight shape changed")
    x0 = reference_positions.detach().clone().requires_grad_(True)
    epsilon0 = epsilon_reference_fn(x0)
    if epsilon0.shape != epsilon_current.shape:
        raise ValueError("reference/current node energy shape changed")
    weighted_reference = torch.dot(live_weight, epsilon0)
    weighted_gradient = torch.autograd.grad(
        weighted_reference, x0, create_graph=True, retain_graph=True
    )[0]
    weighted_hvp = torch.autograd.grad(
        weighted_gradient,
        x0,
        grad_outputs=displacement,
        create_graph=True,
        retain_graph=True,
    )[0]
    return (
        torch.dot(live_weight, epsilon_current)
        - weighted_reference
        - torch.sum(weighted_gradient * displacement)
        - 0.5 * torch.sum(displacement * weighted_hvp)
    )


def _production_energy_components(
    context: _VerifiedProductionContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    aligned = (
        context.current_positions_reference_order - context.image_shift.detach()
    )
    observables = context.node_observables_fn(aligned)
    fields = context.background_fields_fn(aligned)
    fixed = weighted_node_taylor2_carrier_energy(
        observables.scale_shift_node_energy_eV,
        lambda positions: context.node_observables_fn(
            positions
        ).scale_shift_node_energy_eV,
        fields.multipolar_gate,
        context.current_positions_reference_order,
        context.reference_positions,
        context.image_shift,
    )
    parameters = context.parameter_basis_fn(observables, fields)
    if parameters.shape != (LINEAR_DESIGN_WIDTH,):
        raise ValueError("R2R parameter basis width changed")
    return fixed, parameters


def production_linear_design_query(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    *,
    device: torch.device | str,
    create_graph: bool = False,
    formal_graph_data: Mapping[str, torch.Tensor] | None = None,
    graph_mode: str = "baseline",
    baseline_reference_template: Atoms | None = None,
    baseline_structure_template: Atoms | None = None,
    rigid_transform: np.ndarray | None = None,
) -> ProductionLinearDesignQuery:
    """Unique production path from absolute geometry to the FP64 65-column design.

    The 66-column path is intentionally restricted to ``create_graph=False``.
    Complete mechanics must use the scalar-first fast API below; retaining 66
    force graphs before a full Hessian has prohibitive memory amplification.
    """
    if create_graph:
        raise ValueError(
            "66-column create_graph mechanics is forbidden; use the fast combined API"
        )
    context = _verified_production_context(
        model,
        structure,
        reference_template,
        device=device,
        formal_graph_data=formal_graph_data,
        graph_mode=graph_mode,
        baseline_reference_template=baseline_reference_template,
        baseline_structure_template=baseline_structure_template,
        rigid_transform=rigid_transform,
    )
    fixed_energy, parameter_energy = _production_energy_components(context)
    fixed_force_reference = -torch.autograd.grad(
        fixed_energy,
        context.current_positions_reference_order,
        retain_graph=True,
    )[0].reshape(-1, 1)
    parameter_force_columns = []
    for column in range(LINEAR_DESIGN_WIDTH):
        parameter_force_columns.append(
            -torch.autograd.grad(
                parameter_energy[column],
                context.current_positions_reference_order,
                retain_graph=True,
            )[0].reshape(-1)
        )
    parameter_force_reference = torch.stack(parameter_force_columns, dim=1)
    atom_count = context.atom_count
    reference_fixed = fixed_force_reference.reshape(atom_count, 3, 1)
    reference_parameter = parameter_force_reference.reshape(
        atom_count, 3, LINEAR_DESIGN_WIDTH
    )
    source_fixed = torch.empty_like(reference_fixed)
    source_parameter = torch.empty_like(reference_parameter)
    reference_to_source = torch.as_tensor(
        context.assignment.reference_to_source,
        dtype=torch.long,
        device=reference_fixed.device,
    )
    source_fixed[reference_to_source] = reference_fixed
    source_parameter[reference_to_source] = reference_parameter
    return ProductionLinearDesignQuery(
        fixed_offset_energy_eV=fixed_energy.detach().reshape(1),
        fixed_offset_force_source_order_eV_A=source_fixed.reshape(3 * atom_count, 1),
        parameter_energy_design_eV=parameter_energy.detach(),
        parameter_force_design_source_order_eV_A=source_parameter.reshape(
            3 * atom_count, LINEAR_DESIGN_WIDTH
        ),
        assignment=context.assignment,
        receipt=context.receipt,
        _live_current_positions_reference_order=(
            context.current_positions_reference_order
        ),
    )


def production_fixed_carrier_energy_force(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    *,
    device: torch.device | str,
    formal_graph_data: Mapping[str, torch.Tensor] | None = None,
    graph_mode: str = "baseline",
    baseline_reference_template: Atoms | None = None,
    baseline_structure_template: Atoms | None = None,
    rigid_transform: np.ndarray | None = None,
    create_graph: bool = False,
) -> ProductionCombinedProbe:
    """Hash-bound corrected fixed carrier without evaluating 65 force columns."""
    context = _verified_production_context(
        model,
        structure,
        reference_template,
        device=device,
        formal_graph_data=formal_graph_data,
        graph_mode=graph_mode,
        baseline_reference_template=baseline_reference_template,
        baseline_structure_template=baseline_structure_template,
        rigid_transform=rigid_transform,
    )
    fixed, _ = _production_energy_components(context)
    force_reference = -torch.autograd.grad(
        fixed,
        context.current_positions_reference_order,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    return ProductionCombinedProbe(
        energy_eV=fixed,
        force_source_order_eV_A=_source_order_force(
            force_reference, context.assignment
        ),
        coefficients_sha256=FIXED_CARRIER_COEFFICIENT_SHA256,
        query_receipt={
            **context.receipt,
            "mechanics_path": "corrected_live_weighted_fixed_carrier",
            "legacy_raw_a_epsilon_used": False,
            "fixed_carrier_coefficient_sha256": (
                FIXED_CARRIER_COEFFICIENT_SHA256
            ),
        },
    )


@dataclass
class ProductionCombinedMechanics:
    energy_eV: torch.Tensor
    force_source_order_eV_A: torch.Tensor
    Hessian_source_order_eV_A2: torch.Tensor
    Hessian_antisymmetry_max_abs_eV_A2: float
    Hessian_translation_ASR_max_abs_eV_A2: float
    query_receipt: dict
    coefficients_sha256: str


def production_fixed_carrier_energy_force_hessian(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    *,
    device: torch.device | str,
    formal_graph_data: Mapping[str, torch.Tensor] | None = None,
    graph_mode: str = "baseline",
    baseline_reference_template: Atoms | None = None,
    baseline_structure_template: Atoms | None = None,
    rigid_transform: np.ndarray | None = None,
) -> ProductionCombinedMechanics:
    """Hash-bound E/F/full-H audit path for the corrected fixed carrier."""
    context = _verified_production_context(
        model,
        structure,
        reference_template,
        device=device,
        formal_graph_data=formal_graph_data,
        graph_mode=graph_mode,
        baseline_reference_template=baseline_reference_template,
        baseline_structure_template=baseline_structure_template,
        rigid_transform=rigid_transform,
    )
    fixed, _ = _production_energy_components(context)
    force_reference = -torch.autograd.grad(
        fixed,
        context.current_positions_reference_order,
        create_graph=True,
        retain_graph=True,
    )[0]
    reference_rows = []
    for component in force_reference.reshape(-1):
        reference_rows.append(
            -torch.autograd.grad(
                component,
                context.current_positions_reference_order,
                retain_graph=True,
            )[0].reshape(-1)
        )
    reference_hessian = torch.stack(reference_rows)
    reference_to_source = torch.as_tensor(
        context.assignment.reference_to_source,
        dtype=torch.long,
        device=reference_hessian.device,
    )
    source_components = (
        3 * reference_to_source[:, None]
        + torch.arange(3, device=reference_hessian.device)[None, :]
    ).reshape(-1)
    hessian = torch.empty_like(reference_hessian)
    hessian[source_components[:, None], source_components[None, :]] = (
        reference_hessian
    )
    force_source = _source_order_force(force_reference, context.assignment)
    atom_count = context.atom_count
    blocks = hessian.reshape(atom_count, 3, atom_count, 3)
    return ProductionCombinedMechanics(
        energy_eV=fixed,
        force_source_order_eV_A=force_source,
        Hessian_source_order_eV_A2=hessian,
        Hessian_antisymmetry_max_abs_eV_A2=float(
            torch.max(torch.abs(hessian - hessian.T)).detach()
        ),
        Hessian_translation_ASR_max_abs_eV_A2=float(
            torch.max(torch.abs(blocks.sum(dim=2))).detach()
        ),
        query_receipt={
            **context.receipt,
            "mechanics_path": "corrected_live_weighted_fixed_carrier_full_Hessian",
            "legacy_raw_a_epsilon_used": False,
            "fixed_carrier_coefficient_sha256": (
                FIXED_CARRIER_COEFFICIENT_SHA256
            ),
        },
        coefficients_sha256=FIXED_CARRIER_COEFFICIENT_SHA256,
    )


def _canonical_coefficient_tensor(
    coefficients: Sequence[float], *, dtype: torch.dtype, device: torch.device | str
) -> tuple[torch.Tensor, str]:
    coefficient_values = [float(value) for value in coefficients]
    coefficient_hash = semantic_sha256(coefficient_values)
    affine_hash = semantic_sha256([1.0, *coefficient_values])
    if {
        "parameter": coefficient_hash,
        "affine_[1,p]": affine_hash,
    } != {
        "parameter": MECHANICS_PROBE_COEFFICIENTS_SHA256,
        "affine_[1,p]": MECHANICS_AFFINE_COEFFICIENTS_SHA256,
    }:
        raise ValueError("formal mechanics coefficients differ from canonical probe")
    parameter = torch.as_tensor(coefficient_values, dtype=dtype, device=device)
    affine = torch.cat((parameter.new_ones(1), parameter))
    return affine, coefficient_hash


def _taylor_null_combined_energy_force(
    context: _VerifiedProductionContext,
    affine_coefficients: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine the corrected carrier and 65 zero-jet parameter columns."""
    x = context.current_positions_reference_order
    expected_width = LINEAR_DESIGN_WIDTH + 1
    if affine_coefficients.shape != (expected_width,):
        raise ValueError("combined affine coefficient vector must have width 66")
    if not torch.equal(
        affine_coefficients[:1], affine_coefficients.new_ones(1)
    ):
        raise ValueError("corrected fixed carrier coefficient must be exactly one")
    fixed, parameters = _production_energy_components(context)
    remainder = fixed + torch.dot(parameters, affine_coefficients[1:])
    force_reference = -torch.autograd.grad(
        remainder, x, create_graph=create_graph
    )[0]
    return remainder, force_reference


def _source_order_force(
    force_reference_order: torch.Tensor, assignment: Assignment
) -> torch.Tensor:
    source = torch.empty_like(force_reference_order)
    reference_to_source = torch.as_tensor(
        assignment.reference_to_source,
        dtype=torch.long,
        device=force_reference_order.device,
    )
    source[reference_to_source] = force_reference_order
    return source


def production_combined_energy_force(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    *,
    device: torch.device | str,
    coefficients: Sequence[float] = MECHANICS_PROBE_COEFFICIENTS,
    formal_graph_data: Mapping[str, torch.Tensor] | None = None,
    graph_mode: str = "baseline",
    baseline_reference_template: Atoms | None = None,
    baseline_structure_template: Atoms | None = None,
    rigid_transform: np.ndarray | None = None,
    create_graph: bool = False,
) -> ProductionCombinedProbe:
    """Memory-bounded scalar-first production E/F for canonical mechanics."""
    context = _verified_production_context(
        model,
        structure,
        reference_template,
        device=device,
        formal_graph_data=formal_graph_data,
        graph_mode=graph_mode,
        baseline_reference_template=baseline_reference_template,
        baseline_structure_template=baseline_structure_template,
        rigid_transform=rigid_transform,
    )
    affine_coefficients, coefficient_hash = _canonical_coefficient_tensor(
        coefficients,
        dtype=context.current_positions_reference_order.dtype,
        device=context.current_positions_reference_order.device,
    )
    energy, force_reference = _taylor_null_combined_energy_force(
        context, affine_coefficients, create_graph=create_graph
    )
    force_source = _source_order_force(force_reference, context.assignment)
    receipt = {
        **context.receipt,
        "mechanics_path": "corrected_weighted_node_carrier_plus_65_zerojet",
        "affine_combination_order": "corrected_fixed_carrier_plus_parameter_65_dot_p",
        "legacy_66_column_create_graph_used": False,
        "mechanics_coefficients_sha256": coefficient_hash,
        "mechanics_affine_[1,p]_sha256": MECHANICS_AFFINE_COEFFICIENTS_SHA256,
    }
    return ProductionCombinedProbe(
        energy_eV=energy,
        force_source_order_eV_A=force_source,
        coefficients_sha256=coefficient_hash,
        query_receipt=receipt,
    )


def production_rigid_transform_native_rebuild_diagnostic(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    *,
    baseline_reference_template: Atoms,
    baseline_structure_template: Atoms,
    rigid_transform: np.ndarray,
    device: torch.device | str,
    coefficients: Sequence[float] = MECHANICS_PROBE_COEFFICIENTS,
) -> dict:
    """Quantify native-rebuild sensitivity without authorizing formal physics."""
    covariant_context = _verified_production_context(
        model,
        structure,
        reference_template,
        device=device,
        graph_mode="rigid_transform_probe",
        baseline_reference_template=baseline_reference_template,
        baseline_structure_template=baseline_structure_template,
        rigid_transform=rigid_transform,
    )
    native_context = _verified_production_context(
        model,
        structure,
        reference_template,
        device=device,
        graph_mode="rigid_transform_probe",
        baseline_reference_template=baseline_reference_template,
        baseline_structure_template=baseline_structure_template,
        rigid_transform=rigid_transform,
        _diagnostic_native_rebuild=True,
    )
    affine, coefficient_hash = _canonical_coefficient_tensor(
        coefficients,
        dtype=covariant_context.current_positions_reference_order.dtype,
        device=covariant_context.current_positions_reference_order.device,
    )
    covariant_energy, covariant_force_reference = _taylor_null_combined_energy_force(
        covariant_context, affine, create_graph=False
    )
    native_energy, native_force_reference = _taylor_null_combined_energy_force(
        native_context, affine, create_graph=False
    )
    covariant_force = _source_order_force(
        covariant_force_reference, covariant_context.assignment
    )
    native_force = _source_order_force(native_force_reference, native_context.assignment)
    if not np.array_equal(
        covariant_context.assignment.reference_to_source,
        native_context.assignment.reference_to_source,
    ):
        raise RuntimeError("native graph diagnostic changed source mapping")
    return {
        "format": "graphene_r2r_native_rebuild_combined_EF_diagnostic_v1",
        "authorization_role": "diagnostic_only_not_physics_gate",
        "mechanics_coefficients_sha256": coefficient_hash,
        "covariant_energy_eV": float(covariant_energy.detach()),
        "native_rebuilt_energy_eV": float(native_energy.detach()),
        "energy_abs_difference_eV": float(
            torch.abs(native_energy - covariant_energy).detach()
        ),
        "force_max_abs_difference_eV_A": float(
            torch.max(torch.abs(native_force - covariant_force)).detach()
        ),
        "covariant_force_semantic_sha256": _array_semantic_sha256(
            (("force", covariant_force.detach().cpu().numpy().astype(np.float64)),)
        ),
        "native_rebuilt_force_semantic_sha256": _array_semantic_sha256(
            (("force", native_force.detach().cpu().numpy().astype(np.float64)),)
        ),
        "covariant_query_receipt": covariant_context.receipt,
        "native_rebuilt_query_receipt": native_context.receipt,
        "physics_gate_authorized": False,
    }


def production_combined_energy_force_hessian(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    *,
    device: torch.device | str,
    coefficients: Sequence[float] = MECHANICS_PROBE_COEFFICIENTS,
    formal_graph_data: Mapping[str, torch.Tensor] | None = None,
    graph_mode: str = "baseline",
    baseline_reference_template: Atoms | None = None,
    baseline_structure_template: Atoms | None = None,
    rigid_transform: np.ndarray | None = None,
) -> ProductionCombinedMechanics:
    """Only formal full-H path: scalar-first, single-graph Taylor mechanics."""
    context = _verified_production_context(
        model,
        structure,
        reference_template,
        device=device,
        formal_graph_data=formal_graph_data,
        graph_mode=graph_mode,
        baseline_reference_template=baseline_reference_template,
        baseline_structure_template=baseline_structure_template,
        rigid_transform=rigid_transform,
    )
    affine_coefficients, coefficient_hash = _canonical_coefficient_tensor(
        coefficients,
        dtype=context.current_positions_reference_order.dtype,
        device=context.current_positions_reference_order.device,
    )
    energy, force_reference = _taylor_null_combined_energy_force(
        context, affine_coefficients, create_graph=True
    )
    live_reference = context.current_positions_reference_order
    reference_hessian_rows = []
    for component in force_reference.reshape(-1):
        derivative = torch.autograd.grad(
            component, live_reference, retain_graph=True
        )[0]
        reference_hessian_rows.append(-derivative.reshape(-1))
    reference_hessian = torch.stack(reference_hessian_rows)
    atom_count = len(structure)
    hessian = torch.empty_like(reference_hessian)
    reference_to_source = torch.as_tensor(
        context.assignment.reference_to_source,
        dtype=torch.long,
        device=reference_hessian.device,
    )
    source_components = (
        3 * reference_to_source[:, None]
        + torch.arange(3, device=reference_hessian.device)[None, :]
    ).reshape(-1)
    hessian[source_components[:, None], source_components[None, :]] = reference_hessian
    force_source = _source_order_force(force_reference, context.assignment)
    antisymmetry = float(torch.max(torch.abs(hessian - hessian.T)).detach())
    blocks = hessian.reshape(atom_count, 3, atom_count, 3)
    translation_asr = float(torch.max(torch.abs(blocks.sum(dim=2))).detach())
    return ProductionCombinedMechanics(
        energy_eV=energy,
        force_source_order_eV_A=force_source.reshape(atom_count, 3),
        Hessian_source_order_eV_A2=hessian,
        Hessian_antisymmetry_max_abs_eV_A2=antisymmetry,
        Hessian_translation_ASR_max_abs_eV_A2=translation_asr,
        query_receipt={
            **context.receipt,
            "mechanics_path": (
                "corrected_weighted_node_carrier_plus_65_zerojet_full_Hessian"
            ),
            "affine_combination_order": (
                "corrected_fixed_carrier_plus_parameter_65_dot_p"
            ),
            "legacy_66_column_create_graph_used": False,
            "mechanics_coefficients_sha256": coefficient_hash,
            "mechanics_affine_[1,p]_sha256": (
                MECHANICS_AFFINE_COEFFICIENTS_SHA256
            ),
        },
        coefficients_sha256=coefficient_hash,
    )


def rank_condition_precheck(design: np.ndarray) -> dict:
    """Zero-parameter SVD precheck after train-RMS scaling without centering."""
    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != LINEAR_DESIGN_WIDTH:
        raise ValueError("force design must be a row-by-65 matrix")
    if matrix.shape[0] < LINEAR_DESIGN_WIDTH:
        raise ValueError("force design has fewer rows than columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("force design contains non-finite values")
    rms = np.sqrt(np.mean(matrix * matrix, axis=0))
    maximum = float(np.max(rms))
    if maximum == 0.0:
        active = np.zeros(LINEAR_DESIGN_WIDTH, dtype=bool)
    else:
        active = rms > ZERO_COLUMN_RELATIVE_RMS * maximum
    scaled = matrix[:, active] / rms[active] if np.any(active) else matrix[:, :0]
    singular = np.linalg.svd(scaled, full_matrices=False, compute_uv=False)
    if singular.size:
        tolerance = max(scaled.shape) * np.finfo(np.float64).eps * singular[0]
        rank = int(np.sum(singular > tolerance))
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    else:
        tolerance = 0.0
        rank = 0
        condition = None
    full = bool(np.all(active) and rank == LINEAR_DESIGN_WIDTH)
    condition_ok = bool(full and condition is not None and condition <= SCALED_CONDITION_REPORT_LIMIT)
    return {
        "format": "graphene_r2r0_rank_condition_precheck_v1",
        "rows": int(matrix.shape[0]),
        "columns": LINEAR_DESIGN_WIDTH,
        "scaling": "RMS_without_mean_centering",
        "zero_column_relative_RMS_threshold": ZERO_COLUMN_RELATIVE_RMS,
        "column_RMS": rms.tolist(),
        "active_columns": np.flatnonzero(active).tolist(),
        "zero_or_near_zero_columns": np.flatnonzero(~active).tolist(),
        "rank_tolerance": float(tolerance),
        "scaled_rank": rank,
        "scaled_singular_values": singular.tolist(),
        "scaled_condition_number": condition,
        "full_column_rank": full,
        "condition_report_limit": SCALED_CONDITION_REPORT_LIMIT,
        "condition_within_report_limit": condition_ok,
        "R2R0_precheck_pass": condition_ok,
        "pass_definition": "all 65 columns nonzero, scaled rank 65, condition <= 1e8",
        "can_authorize_fit_or_training": False,
    }


class R2R0DesignCollector:
    """Collect design rows only; target arrays are intentionally unsupported."""

    def __init__(self) -> None:
        self._matrices: list[np.ndarray] = []
        self._split_counts = {key: 0 for key in EXPECTED_SPLIT_COUNTS}
        self._identities: set[str] = set()

    def add(self, split: str, identity: str, force_design: np.ndarray) -> None:
        if split not in EXPECTED_SPLIT_COUNTS:
            raise ValueError("R2R-0 split is not train-only")
        if identity in self._identities:
            raise ValueError("duplicate R2R-0 structure identity")
        matrix = np.asarray(force_design, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != LINEAR_DESIGN_WIDTH:
            raise ValueError("one structure force design must have 65 columns")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("one structure force design is non-finite")
        self._identities.add(identity)
        self._split_counts[split] += 1
        self._matrices.append(matrix.copy())

    def finalize(self) -> tuple[np.ndarray, dict]:
        if self._split_counts != EXPECTED_SPLIT_COUNTS:
            raise ValueError(
                f"R2R-0 split counts changed: {self._split_counts}"
            )
        matrix = np.concatenate(self._matrices, axis=0)
        receipt = rank_condition_precheck(matrix)
        receipt["split_counts"] = dict(self._split_counts)
        receipt["design_semantic_sha256"] = semantic_sha256(
            {
                "dtype": str(matrix.dtype),
                "shape": list(matrix.shape),
                "bytes_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
            }
        )
        return matrix, receipt


def validate_r2r0_input_paths(paths: Mapping[str, Path]) -> dict[str, Path]:
    """Lexically reject held/support inputs before any open, resolve, or hash."""
    if set(paths) != set(EXPECTED_INPUTS):
        raise ValueError("R2R-0 input key set changed")
    checked: dict[str, Path] = {}
    for role, candidate in paths.items():
        raw = str(candidate)
        lowered = raw.lower()
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            raise ValueError(f"forbidden R2R-0 path token for {role}")
        pure = PurePath(raw)
        if ".." in pure.parts:
            raise ValueError("R2R-0 path traversal is forbidden")
        path = Path(candidate)
        if path.name != EXPECTED_INPUTS[role]["basename"]:
            raise ValueError(f"unexpected R2R-0 basename for {role}")
        if path.is_symlink():
            raise ValueError("R2R-0 input symlinks are forbidden")
        checked[role] = path
    return checked


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_r2r0_input_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    checked = validate_r2r0_input_paths(paths)
    observed = {role: sha256_file(path) for role, path in checked.items()}
    for role, digest in observed.items():
        if digest != EXPECTED_INPUTS[role]["sha256"]:
            raise ValueError(f"R2R-0 input hash changed for {role}")
    return observed


def continuous_four_fold_assignments() -> dict[str, list[list[int]]]:
    """Frozen future readout folds; R2R-0 records but does not use them."""
    return {
        "E50_seed0": [list(range(5 * fold, 5 * (fold + 1))) for fold in range(4)],
        "T300": [list(range(9 * fold, 9 * (fold + 1))) for fold in range(4)],
        "T600": [list(range(9 * fold, 9 * (fold + 1))) for fold in range(4)],
    }


def harmonic_rank1_gate_receipt(
    *,
    structure_count: int,
    max_cross_product_square_A4: float,
    max_multipolar_gate: float,
    max_taylor_remainder_energy_eV: float,
    max_taylor_remainder_force_eV_A: float,
) -> dict:
    """Separate actual-value gate; it never accepts or returns design rows."""
    values = (
        max_cross_product_square_A4,
        max_multipolar_gate,
        max_taylor_remainder_energy_eV,
        max_taylor_remainder_force_eV_A,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("harmonic rank-1 gate values must be finite and nonnegative")
    checks = {
        "count_32": structure_count == 32,
        "direct_b_le_1e-20_A4": max_cross_product_square_A4 <= RANK1_NUMERICAL_B_GATE_A4,
        "direct_a_le_1e-12": max_multipolar_gate <= RANK1_NUMERICAL_A_GATE,
        "Taylor_remainder_E_le_1e-10_eV": max_taylor_remainder_energy_eV <= 1.0e-10,
        "Taylor_remainder_F_le_1e-9_eV_A": max_taylor_remainder_force_eV_A <= 1.0e-9,
    }
    return {
        "format": "graphene_r2r0_harmonic_rank1_gate_v1",
        "structure_count": int(structure_count),
        "max_cross_product_square_A4": float(max_cross_product_square_A4),
        "max_multipolar_gate": float(max_multipolar_gate),
        "max_Taylor_remainder_energy_eV": float(max_taylor_remainder_energy_eV),
        "max_Taylor_remainder_force_eV_A": float(max_taylor_remainder_force_eV_A),
        "checks": checks,
        "pass": all(checks.values()),
        "enters_design_scaler_rank_OOF_or_fit": False,
        "can_authorize_fit_or_training": False,
    }


__all__ = [
    "AFFINE_COMPONENT_NAMES",
    "AFFINE_COMPONENT_NAMES_SHA256",
    "BACKGROUND_EFFECTIVE_RADIUS_A",
    "BACKGROUND_INTERACTION_DIAMETER_A",
    "BackgroundFields",
    "CANONICAL_CONTRACT",
    "CANONICAL_CONTRACT_SHA256",
    "EXPECTED_INPUTS",
    "EXPECTED_BASELINE_FORMAL_GRAPH_SHA256",
    "LINEAR_DESIGN_WIDTH",
    "FIXED_CARRIER_COEFFICIENT_SHA256",
    "MECHANICS_PROBE_COEFFICIENTS",
    "MECHANICS_PROBE_COEFFICIENTS_SHA256",
    "MECHANICS_AFFINE_COEFFICIENTS_SHA256",
    "MULTIPOLAR_BETA_A4",
    "OrderedBackground",
    "ProductionCombinedMechanics",
    "ProductionCombinedProbe",
    "ProductionLinearDesignQuery",
    "R2QNodeObservables",
    "R2R0DesignCollector",
    "REFERENCE_CUTOFF_A",
    "ReferenceNeighborhood",
    "RIGID_INVARIANT_GRAPH_TENSOR_KEYS",
    "RIGID_INTERNAL_COVARIANCE_ATOL_A",
    "RIGID_VECTOR_GRAPH_TENSOR_KEYS",
    "SCALE_DERIVATION",
    "SCALE_DERIVATION_SHA256",
    "TRACE_SCALE_A2",
    "TaylorNullLinearDesign",
    "background_from_structure",
    "background_rank0_zero_jet_audit",
    "harmonic_rank1_gate_receipt",
    "continuous_four_fold_assignments",
    "even_l0_channels",
    "fixed_reference_neighborhood",
    "full_structure_semantic_sha256",
    "multipolar_background",
    "native_cell_wrap_order_mic_receipt",
    "quintic_reference_weight",
    "quintic_inside_cutoff_metrics",
    "r2q_node_observables",
    "rank_condition_precheck",
    "parameter_energy_basis",
    "taylor_null_linear_force_design",
    "production_linear_design_query",
    "production_fixed_carrier_energy_force",
    "production_fixed_carrier_energy_force_hessian",
    "production_combined_energy_force",
    "production_combined_energy_force_hessian",
    "production_rigid_transform_native_rebuild_diagnostic",
    "reference_neighborhood_semantic_sha256",
    "reference_neighborhood_physical_multiset_receipt",
    "rigid_transform_native_rebuild_diagnostic",
    "weighted_node_taylor2_carrier_energy",
    "validate_r2r0_input_paths",
    "verify_r2r0_input_hashes",
]
