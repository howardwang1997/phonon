from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read
from e3nn import o3
from mace.modules import ScaleShiftMACE
from mace.modules.blocks import (
    PolynomialCutoff,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))

from graphene_r2o_taylor_null import (  # noqa: E402
    EXPECTED_R_MAX_A,
    adapt_reference_cell,
    fixed_reference_graph,
    mace_interaction_energy,
    reordered_structure,
    solve_assignment,
    source_order_forces,
    validate_mace_architecture,
    whole_energy_taylor2_remainder,
)


DATA = ROOT / "data" / "graphene_r2o_taylor_null_core"


def _hessian_from_force(force: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    rows = []
    for value in force.reshape(-1):
        rows.append(
            -torch.autograd.grad(value, position, retain_graph=True)[0].reshape(-1)
        )
    return torch.stack(rows)


def test_data_freeze_is_thermal_only_pretrain_and_small_harmonic_stage2() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["counts"]["gradient_train_thermal_only"] == 92
    assert manifest["counts"]["valid_e50_seed1_gate_only"] == 20
    assert manifest["counts"]["harmonic_lambda1_small_gate"] == 12
    assert manifest["counts"]["harmonic_full_report_only"] == 25
    assert manifest["counts"]["seed2_opened"] == 0
    assert manifest["counts"]["support_opened"] == 0
    assert manifest["leakage_control"]["E50_seed1_in_gradients_or_scales"] is False
    assert manifest["leakage_control"]["E50_seed2_file_opened"] is False
    assert manifest["leakage_control"]["support_file_opened"] is False
    assert manifest["architecture_freeze"]["r_max_A"] == 3.2
    assert manifest["architecture_freeze"]["cutoff_polynomial_order_p"] == 5
    assert manifest["architecture_freeze"]["interaction_diameter_bound_A"] == 12.8
    assert manifest["fixed_endpoint_thresholds"][
        "harmonic_small_force_RMSE_meV_A"
    ] == 0.5
    assert manifest["fixed_endpoint_thresholds"][
        "harmonic_small_force_max_abs_meV_A"
    ] == 10.0
    assert manifest["fixed_endpoint_thresholds"][
        "Taylor_remainder_reference_Hessian_ASR_row_sum_max_abs_eV_A2"
    ] == 1.0e-7
    assert manifest["training_data_contract"]["fixed_group_mass"] == {
        "E50_seed0": 0.5,
        "T300": 0.25,
        "T600": 0.25,
    }
    assert manifest["training_data_contract"]["stage2_group_mass"] == {
        "E50_seed0": 0.5,
        "T300": 0.2,
        "T600": 0.2,
        "harmonic_lambda1_small_zero": 0.1,
    }
    assert manifest["architecture_freeze"]["pair_repulsion"] is False
    provenance = manifest["training_data_contract"]["target_provenance"]
    assert provenance["E50_identity_max_abs_error_eV_A"] <= 1.1e-8
    assert provenance["auxiliary_identity_max_abs_error_eV_A"] <= 2.0e-15
    assert provenance["all92_are_fixed_smearing_DFT_labels"] is False
    assert provenance["auxiliary_operator_scope"]["T300_degauss_Ry"] == pytest.approx(
        0.0019000869380739254
    )
    assert provenance["auxiliary_operator_scope"]["T600_degauss_Ry"] == pytest.approx(
        0.003800173876147851
    )
    assert manifest["harmonic_gate_contract"][
        "small12_maximum_MIC_atom_displacement_A"
    ] <= 0.03 + manifest["harmonic_gate_contract"][
        "small12_atom_displacement_numerical_tolerance_A"
    ]

    zero_frames = read(DATA / "train_harmonic_lambda1_small_zero.xyz", index=":")
    harmonic_contract = manifest["harmonic_gate_contract"]
    threshold = harmonic_contract["bond_length_RMS_max_A"]
    recomputed_count = sum(
        value <= threshold
        for value in harmonic_contract["all72_harmonic_train_bond_length_RMS_A"]
    )
    assert len(zero_frames) == recomputed_count
    assert manifest["counts"]["stage2_gradient_harmonic_lambda1_small_zero"] == recomputed_count
    assert all(float(item.info["r2o_bond_length_RMS_A"]) <= threshold for item in zero_frames)
    assert all(np.count_nonzero(item.arrays["REF_forces"]) == 0 for item in zero_frames)
    assert all("ORIGINAL_SHORT_DELTA_REF_forces" in item.arrays for item in zero_frames)


def test_phonopy_reference_assignment_identity_unique_and_permutation_safe() -> None:
    structure = read(DATA / "train_thermal.xyz", index=0)
    template = read(DATA / "reference_6x6.xyz", index=0)
    reference = adapt_reference_cell(template, structure)
    identity = solve_assignment(structure, reference)
    assert np.array_equal(identity.reference_to_source, np.arange(72))
    assert identity.maximum_distance_A < 0.5 * 1.421
    assert identity.minimum_uniqueness_gap_A > 0.9

    generator = np.random.default_rng(83)
    permutation = generator.permutation(len(structure))
    permuted = structure[permutation]
    permuted.set_cell(structure.cell, scale_atoms=False)
    permuted.pbc = structure.pbc
    assignment = solve_assignment(permuted, reference)
    restored = reordered_structure(permuted, assignment)
    np.testing.assert_allclose(restored.positions, structure.positions, atol=1.0e-12)

    force_reference = torch.arange(216, dtype=torch.float64).reshape(72, 3)
    force_source = source_order_forces(force_reference, assignment)
    np.testing.assert_array_equal(
        force_source.detach().numpy()[assignment.reference_to_source],
        force_reference.detach().numpy(),
    )


def test_mic_integer_is_frozen_but_live_displacement_has_identity_jacobian() -> None:
    structure = read(DATA / "train_thermal.xyz", index=0)
    template = read(DATA / "reference_6x6.xyz", index=0)
    shifted = structure.copy()
    shifted.positions[7] += np.asarray(structure.cell[0])
    reference = adapt_reference_cell(template, shifted)
    assignment = solve_assignment(shifted, reference)
    assert assignment.image_integer_reference_order[7, 0] == 1

    x = torch.tensor(
        shifted.positions[assignment.reference_to_source],
        dtype=torch.float64,
        requires_grad=True,
    )
    x0 = torch.tensor(reference.positions, dtype=torch.float64)
    cell = torch.tensor(np.asarray(reference.cell), dtype=torch.float64)
    image = torch.tensor(assignment.image_integer_reference_order, dtype=torch.float64)
    u = x - x0 - image @ cell
    direction = torch.randn_like(u)
    derivative = torch.autograd.grad((u * direction).sum(), x)[0]
    torch.testing.assert_close(derivative, direction)


def test_whole_energy_remainder_removes_exact_constant_linear_quadratic_jet() -> None:
    dtype = torch.float64
    x0 = torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=dtype)
    linear = torch.tensor([[0.7, -0.4], [0.2, 0.6]], dtype=dtype)
    matrix = torch.tensor(
        [
            [2.0, 0.2, -0.1, 0.0],
            [0.2, 1.4, 0.3, -0.2],
            [-0.1, 0.3, 1.1, 0.4],
            [0.0, -0.2, 0.4, 0.9],
        ],
        dtype=dtype,
    )

    def energy(position: torch.Tensor) -> torch.Tensor:
        displacement = (position - x0).reshape(-1)
        cubic = 0.13 * displacement.pow(3).sum()
        quartic = 0.07 * displacement.pow(4).sum()
        return (
            torch.tensor([1.9], dtype=dtype)
            + (linear * (position - x0)).sum().reshape(1)
            + (0.5 * displacement @ matrix @ displacement).reshape(1)
            + cubic.reshape(1)
            + quartic.reshape(1)
        )

    reference_live = x0.clone().requires_grad_(True)
    at_reference = whole_energy_taylor2_remainder(
        energy,
        reference_live,
        x0,
        torch.zeros_like(x0),
        create_graph=True,
    )
    assert abs(float(at_reference.energy)) < 1.0e-14
    assert float(at_reference.forces_reference_order.abs().max()) < 1.0e-14
    hessian = _hessian_from_force(
        at_reference.forces_reference_order, reference_live
    )
    assert float(hessian.abs().max()) < 2.0e-13

    displacement = torch.tensor(
        [[0.08, -0.04], [0.02, 0.05]], dtype=dtype
    )
    current = (x0 + displacement).clone().requires_grad_(True)
    observed = whole_energy_taylor2_remainder(
        energy, current, x0, torch.zeros_like(x0), create_graph=True
    )
    expected = 0.13 * displacement.pow(3).sum() + 0.07 * displacement.pow(4).sum()
    torch.testing.assert_close(observed.energy.squeeze(), expected, atol=1.0e-13, rtol=0.0)


def test_detaching_hvp_direction_loses_half_the_quadratic_counterforce() -> None:
    dtype = torch.float64
    reference = torch.zeros((1, 1), dtype=dtype)
    current = torch.tensor([[0.2]], dtype=dtype, requires_grad=True)
    x0 = reference.clone().requires_grad_(True)
    u = current - x0.detach()
    energy_current = 0.5 * 3.0 * current.square().sum()
    energy_reference = 0.5 * 3.0 * x0.square().sum()
    gradient = torch.autograd.grad(energy_reference, x0, create_graph=True)[0]
    wrong_hu = torch.autograd.grad(
        gradient, x0, grad_outputs=u.detach(), create_graph=True
    )[0]
    wrong = (
        energy_current
        - energy_reference
        - (gradient * u).sum()
        - 0.5 * (u * wrong_hu).sum()
    )
    wrong_force = -torch.autograd.grad(wrong, current)[0]
    torch.testing.assert_close(wrong_force, torch.tensor([[-0.3]], dtype=dtype))


def test_p5_cutoff_is_c2_but_not_c3_at_rmax() -> None:
    radius = torch.tensor([EXPECTED_R_MAX_A], dtype=torch.float64, requires_grad=True)
    p = torch.tensor(5.0, dtype=torch.float64)
    reduced = radius / EXPECTED_R_MAX_A
    value = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * reduced**p
        + p * (p + 2.0) * reduced ** (p + 1.0)
        - (p * (p + 1.0) / 2.0) * reduced ** (p + 2.0)
    )
    first = torch.autograd.grad(value.sum(), radius, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), radius, create_graph=True)[0]
    third = torch.autograd.grad(second.sum(), radius)[0]
    assert abs(float(value)) < 1.0e-14
    assert abs(float(first)) < 1.0e-13
    assert abs(float(second)) < 1.0e-12
    assert abs(float(third)) > 1.0e-3
    cutoff = PolynomialCutoff(r_max=EXPECTED_R_MAX_A, p=5)
    values = [
        float(cutoff(torch.tensor([EXPECTED_R_MAX_A * (1.0 - gap)], dtype=torch.float64)))
        for gap in (1.0e-2, 1.0e-3, 1.0e-4)
    ]
    assert values[0] > values[1] > values[2] > 0.0
    assert 500.0 < values[0] / values[1] < 2000.0
    assert 500.0 < values[1] / values[2] < 2000.0


def test_nontrivial_scaleshift_manual_energy_and_position_gradient_match_forward() -> None:
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = ScaleShiftMACE(
            r_max=EXPECTED_R_MAX_A,
            num_bessel=24,
            num_polynomial_cutoff=5,
            max_ell=2,
            interaction_cls=RealAgnosticResidualInteractionBlock,
            interaction_cls_first=RealAgnosticInteractionBlock,
            num_interactions=2,
            num_elements=1,
            hidden_irreps=o3.Irreps("16x0e+16x1o+16x2e"),
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.asarray([0.0]),
            avg_num_neighbors=12.0,
            atomic_numbers=[6],
            correlation=3,
            gate=torch.nn.functional.silu,
            pair_repulsion=False,
            atomic_inter_scale=2.25,
            atomic_inter_shift=-0.375,
            heads=["Default"],
        ).double()
    finally:
        torch.set_default_dtype(previous_dtype)
    architecture = validate_mace_architecture(model)
    assert architecture["pair_repulsion"] is False
    assert architecture["max_ell"] == 2
    assert architecture["spherical_harmonics_irreps"] == "1x0e+1x1o+1x2e"
    assert architecture["interaction_classes"] == [
        "RealAgnosticInteractionBlock",
        "RealAgnosticResidualInteractionBlock",
    ]
    assert architecture["product_correlations"] == [3, 3, 3, 3]
    assert float(model.scale_shift.scale) == pytest.approx(2.25)
    assert float(model.scale_shift.shift) == pytest.approx(-0.375)

    reference = read(DATA / "reference_6x6.xyz", index=0)
    graph = fixed_reference_graph(reference, device="cpu", dtype=torch.float64)
    positions = torch.tensor(
        np.asarray(reference.positions, float), dtype=torch.float64, requires_grad=True
    )
    positions.data[0, 2] += 0.027
    manual = mace_interaction_energy(model, graph, positions)
    official_data = dict(graph)
    official_data["positions"] = positions
    official = model(
        official_data, training=False, compute_force=False
    )["interaction_energy"]
    manual_gradient = torch.autograd.grad(
        manual.sum(), positions, retain_graph=True
    )[0]
    official_gradient = torch.autograd.grad(official.sum(), positions)[0]
    torch.testing.assert_close(manual, official, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(
        manual_gradient, official_gradient, atol=1.0e-11, rtol=0.0
    )


@pytest.mark.parametrize("reflection", [False, True])
def test_whole_energy_taylor_remainder_is_o3_covariant(reflection: bool) -> None:
    dtype = torch.float64
    reference = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.2], [0.2, 1.1, -0.1]], dtype=dtype
    )
    current0 = reference + torch.tensor(
        [[0.02, -0.01, 0.03], [-0.04, 0.02, 0.01], [0.01, 0.03, -0.02]],
        dtype=dtype,
    )

    def pair_energy(position: torch.Tensor) -> torch.Tensor:
        distances2 = []
        for left in range(3):
            for right in range(left + 1, 3):
                distances2.append((position[left] - position[right]).square().sum())
        values = torch.stack(distances2)
        return (0.2 * values + 0.07 * values.square()).sum().reshape(1)

    current = current0.clone().requires_grad_(True)
    original = whole_energy_taylor2_remainder(
        pair_energy, current, reference, torch.zeros_like(reference), create_graph=False
    )
    generator = np.random.default_rng(4)
    q, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if reflection:
        q[:, 0] *= -np.linalg.det(q)
        if np.linalg.det(q) > 0:
            q[:, 0] *= -1
    elif np.linalg.det(q) < 0:
        q[:, 0] *= -1
    rotation = torch.tensor(q, dtype=dtype)
    rotated_reference = reference @ rotation.T
    rotated_current = (current0 @ rotation.T).clone().requires_grad_(True)
    transformed = whole_energy_taylor2_remainder(
        pair_energy,
        rotated_current,
        rotated_reference,
        torch.zeros_like(reference),
        create_graph=False,
    )
    torch.testing.assert_close(transformed.energy, original.energy, atol=2.0e-13, rtol=0.0)
    torch.testing.assert_close(
        transformed.forces_reference_order,
        original.forces_reference_order @ rotation.T,
        atol=3.0e-12,
        rtol=0.0,
    )
