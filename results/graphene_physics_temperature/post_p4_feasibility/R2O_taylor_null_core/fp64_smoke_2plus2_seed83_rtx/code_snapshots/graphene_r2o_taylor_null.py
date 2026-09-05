#!/usr/bin/env python3
"""Live whole-energy Taylor-null wrapper for the R2O graphene core.

For one local MACE interaction energy ``E_theta`` and the order-matched
pristine reference ``x0``, R2O deploys exactly

    R_theta(x) = E_theta(x) - E_theta(x0)
                 - D E_theta(x0)[u] - 1/2 D2 E_theta(x0)[u,u],

where ``u`` is a live Cartesian minimum-image displacement.  The discrete
minimum-image integers and atom assignment are solved outside autograd; once
frozen, ``du/dx = I``.  The HVP receives the *live* ``u`` as ``grad_outputs``.
Detaching it would remove half of the quadratic counter-force and is forbidden.

There is no force shortcut in this module.  Forces and Hessians are always
autograd derivatives of the complete scalar remainder energy.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from mace.data import AtomicData, KeySpecification, config_from_atoms
from mace.modules.utils import get_edge_vectors_and_lengths
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.scatter import scatter_sum
from scipy.optimize import linear_sum_assignment


FORMAT = "graphene_r2o_whole_energy_taylor2_v1"
EXPECTED_R_MAX_A = 3.2
EXPECTED_NUM_INTERACTIONS = 2
EXPECTED_CUTOFF_P = 5
EXPECTED_MAX_ELL = 2
EXPECTED_CORRELATION = 3
EXPECTED_INTERACTION_CLASSES = (
    "RealAgnosticInteractionBlock",
    "RealAgnosticResidualInteractionBlock",
)
EXPECTED_PAIR_REPULSION = False
EXPECTED_RAW_NODE_WIDTH = 160
EXPECTED_HIDDEN_LAYOUT = (
    (0, 16, 0, 1),
    (0, 16, 1, -1),
    (0, 16, 2, 1),
    (1, 16, 0, 1),
)
INTERACTION_DIAMETER_BOUND_A = 12.8
MINIMUM_CELL_LENGTH_A = 14.76
ASSIGNMENT_MAX_FRACTION_OF_NEAREST_BOND = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def model_dtype(model: torch.nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        return parameter.dtype
    raise ValueError("MACE model has no parameters")


def validate_mace_architecture(model: torch.nn.Module) -> dict:
    r_max = float(model.r_max)
    if abs(r_max - EXPECTED_R_MAX_A) > 1.0e-7:
        raise ValueError(f"R2O requires r_max={EXPECTED_R_MAX_A}; observed {r_max}")
    interactions = int(model.num_interactions)
    if interactions != EXPECTED_NUM_INTERACTIONS:
        raise ValueError("R2O interaction count changed")
    cutoff = model.radial_embedding.cutoff_fn
    cutoff_p = int(cutoff.p)
    if cutoff_p != EXPECTED_CUTOFF_P:
        raise ValueError("R2O requires the frozen p=5 polynomial cutoff")
    spherical_harmonics_irreps = str(model.spherical_harmonics.irreps_out)
    expected_spherical_harmonics_irreps = "1x0e+1x1o+1x2e"
    if spherical_harmonics_irreps != expected_spherical_harmonics_irreps:
        raise ValueError(
            "R2O requires max_ell=2 spherical harmonics; "
            f"observed {spherical_harmonics_irreps}"
        )
    interaction_classes = tuple(
        type(interaction).__name__ for interaction in model.interactions
    )
    if interaction_classes != EXPECTED_INTERACTION_CLASSES:
        raise ValueError(
            "R2O interaction classes changed: "
            f"observed {interaction_classes}"
        )
    correlations = tuple(
        int(contraction.correlation)
        for product in model.products
        for contraction in product.symmetric_contractions.contractions
    )
    if not correlations or any(
        correlation != EXPECTED_CORRELATION for correlation in correlations
    ):
        raise ValueError(
            f"R2O requires correlation=3 throughout; observed {correlations}"
        )
    layout = []
    width = 0
    for interaction, product in enumerate(model.products):
        for multiplicity, irrep in product.linear.irreps_out:
            item = (
                interaction,
                int(multiplicity),
                int(irrep.l),
                int(irrep.p),
            )
            layout.append(item)
            width += int(multiplicity) * int(irrep.dim)
    if tuple(layout) != EXPECTED_HIDDEN_LAYOUT or width != EXPECTED_RAW_NODE_WIDTH:
        raise ValueError(
            f"R2O hidden layout changed: layout={tuple(layout)}, width={width}"
        )
    radial_basis = int(model.radial_embedding.bessel_fn.bessel_weights.numel())
    if radial_basis != 24:
        raise ValueError(f"R2O requires 24 radial basis functions, got {radial_basis}")
    if model_dtype(model) != torch.float64:
        raise ValueError("formal R2O is frozen to float64 for Taylor cancellation")
    if hasattr(model, "pair_repulsion"):
        raise ValueError("formal R2O freezes pair_repulsion=false")
    return {
        "r_max_A": r_max,
        "num_interactions": interactions,
        "cutoff_p": cutoff_p,
        "max_ell": EXPECTED_MAX_ELL,
        "spherical_harmonics_irreps": spherical_harmonics_irreps,
        "interaction_classes": list(interaction_classes),
        "product_correlations": list(correlations),
        "raw_node_width": width,
        "hidden_layout": [list(item) for item in layout],
        "num_radial_basis": radial_basis,
        "dtype": str(model_dtype(model)),
        "pair_repulsion": EXPECTED_PAIR_REPULSION,
        "interaction_diameter_bound_A": INTERACTION_DIAMETER_BOUND_A,
    }


def _canonical_edge_key(i: int, j: int, shift: np.ndarray) -> tuple:
    forward = (int(i), int(j), *(int(value) for value in shift))
    inverse = (int(j), int(i), *(-int(value) for value in shift))
    return min(forward, inverse)


def undirected_edges(structure: Atoms, cutoff: float) -> set[tuple]:
    left, right, shifts = neighbor_list(
        "ijS", structure, cutoff, self_interaction=False
    )
    return {
        _canonical_edge_key(i, j, shift)
        for i, j, shift in zip(left, right, shifts, strict=True)
    }


def nearest_bond_length(reference: Atoms) -> float:
    _, _, distance = neighbor_list(
        "ijd", reference, 1.8, self_interaction=False
    )
    if len(distance) != 3 * len(reference):
        raise ValueError("reference no longer has exactly three nearest neighbours")
    return float(np.min(distance))


@dataclass(frozen=True)
class Assignment:
    """Mapping and frozen image gauge from source order to reference order."""

    reference_to_source: np.ndarray
    source_to_reference: np.ndarray
    image_integer_reference_order: np.ndarray
    maximum_distance_A: float
    minimum_uniqueness_gap_A: float

    def validate(self, count: int) -> None:
        expected = np.arange(count)
        if self.reference_to_source.shape != (count,):
            raise ValueError("reference_to_source has wrong shape")
        if self.source_to_reference.shape != (count,):
            raise ValueError("source_to_reference has wrong shape")
        if self.image_integer_reference_order.shape != (count, 3):
            raise ValueError("MIC image array has wrong shape")
        if not np.array_equal(
            self.source_to_reference[self.reference_to_source], expected
        ):
            raise ValueError("source/reference mappings are not inverse")
        if self.minimum_uniqueness_gap_A <= 0.0:
            raise ValueError("atom assignment is not unique")


def adapt_reference_cell(template: Atoms, structure: Atoms) -> Atoms:
    """Rotate/reflect a frozen fractional reference with an equal-metric cell."""
    if len(template) != len(structure):
        raise ValueError("reference/current atom count differs")
    template_cell = np.asarray(template.cell, float)
    current_cell = np.asarray(structure.cell, float)
    template_metric = template_cell @ template_cell.T
    current_metric = current_cell @ current_cell.T
    if not np.allclose(template_metric, current_metric, atol=2.0e-6, rtol=0.0):
        raise ValueError("R2O fixed-lattice cell metric changed")
    reference = template.copy()
    fractional = np.asarray(template.get_scaled_positions(wrap=False), float)
    reference.set_cell(current_cell, scale_atoms=False)
    reference.set_scaled_positions(fractional)
    reference.pbc = np.asarray(structure.pbc, bool)
    reference.wrap()
    if float(np.min(reference.cell.lengths()[:2])) + 1.0e-8 < MINIMUM_CELL_LENGTH_A:
        raise ValueError("R2O cell is too small for the force-diameter bound")
    return reference


def solve_assignment(structure: Atoms, reference: Atoms) -> Assignment:
    """Species-aware periodic Hungarian mapping plus frozen MIC integers."""
    if len(structure) != len(reference):
        raise ValueError("assignment atom count differs")
    cell = np.asarray(reference.cell, float)
    inverse_cell = np.linalg.inv(cell)
    raw = (
        np.asarray(structure.positions, float)[:, None, :]
        - np.asarray(reference.positions, float)[None, :, :]
    )
    fractional_raw = raw @ inverse_cell
    image = np.zeros_like(fractional_raw)
    fractional = fractional_raw.copy()
    for axis, periodic in enumerate(reference.pbc):
        if periodic:
            image[:, :, axis] = np.rint(fractional_raw[:, :, axis])
            fractional[:, :, axis] -= image[:, :, axis]
    distance = np.linalg.norm(fractional @ cell, axis=-1)
    incompatible = structure.numbers[:, None] != reference.numbers[None, :]
    cost = np.where(incompatible, 1.0e9, distance)
    source_index, reference_index = linear_sum_assignment(cost)
    count = len(reference)
    reference_to_source = np.empty(count, dtype=int)
    reference_to_source[reference_index] = source_index
    source_to_reference = np.empty(count, dtype=int)
    source_to_reference[source_index] = reference_index
    sorted_distance = np.sort(distance, axis=1)
    nearest = np.argmin(distance, axis=1)
    gap = sorted_distance[:, 1] - sorted_distance[:, 0]
    if not np.array_equal(nearest, source_to_reference):
        raise ValueError("global atom assignment is not each atom's unique nearest site")
    minimum_gap = float(np.min(gap))
    maximum = float(
        np.max(distance[reference_to_source, np.arange(count)])
    )
    if minimum_gap <= 1.0e-8:
        raise ValueError("source/reference assignment is ambiguous")
    if maximum >= (
        ASSIGNMENT_MAX_FRACTION_OF_NEAREST_BOND
        * nearest_bond_length(reference)
    ):
        raise ValueError("source/reference assignment exceeds half a C-C bond")
    image_reference = image[
        reference_to_source, np.arange(count)
    ].astype(np.int64)
    result = Assignment(
        reference_to_source=reference_to_source,
        source_to_reference=source_to_reference,
        image_integer_reference_order=image_reference,
        maximum_distance_A=maximum,
        minimum_uniqueness_gap_A=minimum_gap,
    )
    result.validate(count)
    return result


def reordered_structure(structure: Atoms, assignment: Assignment) -> Atoms:
    result = structure[assignment.reference_to_source].copy()
    result.set_cell(structure.cell, scale_atoms=False)
    result.pbc = structure.pbc
    result.info = dict(structure.info)
    return result


def _dummy_labelled(reference: Atoms) -> Atoms:
    result = reference.copy()
    result.calc = None
    result.info["REF_energy"] = 0.0
    result.arrays["REF_forces"] = np.zeros((len(result), 3), float)
    return result


def fixed_reference_graph(
    reference: Atoms, device: torch.device | str, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    specification = KeySpecification(
        info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"}
    )
    configuration = config_from_atoms(
        _dummy_labelled(reference),
        key_specification=specification,
        config_type_weights=None,
        head_name="Default",
    )
    record = AtomicData.from_config(
        configuration,
        z_table=AtomicNumberTable([6]),
        cutoff=EXPECTED_R_MAX_A,
        heads=["Default"],
    )
    loader = torch_geometric.dataloader.DataLoader(
        [record], batch_size=1, shuffle=False, drop_last=False
    )
    batch = next(iter(loader)).to(device)
    data = batch.to_dict()
    for key, value in list(data.items()):
        if torch.is_tensor(value) and value.is_floating_point():
            data[key] = value.to(dtype=dtype)
    expected_directed_edges = {72: 864, 128: 1536}.get(len(reference))
    if expected_directed_edges is None:
        raise ValueError("R2O reference graph supports only 6x6/8x8")
    if int(data["edge_index"].shape[1]) != expected_directed_edges:
        raise ValueError(
            f"R2O reference edge count changed: {data['edge_index'].shape[1]}"
        )
    return data


def validate_current_edge_set(current_ordered: Atoms, reference: Atoms) -> None:
    """Fail closed if any third/fourth-shell edge crosses r_max=3.2 A."""
    reference_edges = undirected_edges(reference, EXPECTED_R_MAX_A)
    current_edges = undirected_edges(current_ordered, EXPECTED_R_MAX_A)
    if current_edges != reference_edges:
        missing = len(reference_edges - current_edges)
        entered = len(current_edges - reference_edges)
        raise ValueError(
            f"R2O r_max edge topology changed (missing={missing}, entered={entered})"
        )


def mace_interaction_energy(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    positions: torch.Tensor,
) -> torch.Tensor:
    """Functor-free MACE interaction energy; no internal requires_grad_ call."""
    vectors, lengths = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=data["edge_index"],
        shifts=data["shifts"],
    )
    node_heads = data["head"][data["batch"]].to(torch.int64)
    node_indices = torch.arange(
        positions.shape[0], device=positions.device, dtype=torch.int64
    )
    graphs = int(data["ptr"].numel() - 1)
    if hasattr(model, "joint_embedding"):
        raise ValueError("formal R2O does not allow unfrozen auxiliary embeddings")
    if not hasattr(model, "scale_shift"):
        raise ValueError("formal R2O requires ScaleShiftMACE interaction semantics")
    node_feats = model.node_embedding(data["node_attrs"])
    edge_attrs = model.spherical_harmonics(vectors)
    edge_feats, cutoff = model.radial_embedding(
        lengths, data["node_attrs"], data["edge_index"], model.atomic_numbers
    )
    features = []
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
        node_feats = product(
            node_feats=node_feats, sc=skip, node_attrs=data["node_attrs"]
        )
        features.append(node_feats)
    if hasattr(model, "pair_repulsion"):
        # validate_mace_architecture rejects this, but keep this local guard so a
        # caller cannot bypass the formal pair policy by using this helper alone.
        raise ValueError("formal R2O freezes pair_repulsion=false")
    # Match ScaleShiftMACE: the zero pair term enters before scale_shift.
    pair_node_energy = positions.new_zeros((positions.shape[0],))
    node_energy_terms = [pair_node_energy]
    for readout_index, readout in enumerate(model.readouts):
        feature_index = -1 if len(model.readouts) == 1 else readout_index
        node_energy = readout(features[feature_index], node_heads)[
            node_indices, node_heads
        ]
        node_energy_terms.append(node_energy)
    node_interaction_energy = torch.sum(
        torch.stack(node_energy_terms, dim=0), dim=0
    )
    node_interaction_energy = model.scale_shift(
        node_interaction_energy, node_heads
    )
    return scatter_sum(
        node_interaction_energy, data["batch"], dim=-1, dim_size=graphs
    )


@dataclass
class TaylorResult:
    energy: torch.Tensor
    forces_reference_order: torch.Tensor
    raw_energy: torch.Tensor
    reference_energy: torch.Tensor
    linear_term: torch.Tensor
    quadratic_term: torch.Tensor
    displacement: torch.Tensor
    current_positions_reference_order: torch.Tensor


def whole_energy_taylor2_remainder(
    energy_fn,
    current_positions: torch.Tensor,
    reference_positions: torch.Tensor,
    image_shift: torch.Tensor,
    *,
    create_graph: bool,
) -> TaylorResult:
    """Generic exact directional Taylor2 subtraction used by the MACE wrapper."""
    x = current_positions
    if not x.requires_grad:
        raise ValueError("live current positions must require gradients")
    if image_shift.requires_grad:
        raise ValueError("discrete MIC image shift must be detached")
    x0 = reference_positions.detach().clone().requires_grad_(True)
    aligned = x - image_shift.detach()
    displacement = aligned - x0.detach()
    raw_energy = energy_fn(aligned)
    reference_energy = energy_fn(x0)
    reference_gradient = torch.autograd.grad(
        reference_energy.sum(), x0, create_graph=True, retain_graph=True
    )[0]
    # Deliberately live.  A detach here removes half the quadratic force.
    hessian_times_u = torch.autograd.grad(
        reference_gradient,
        x0,
        grad_outputs=displacement,
        create_graph=True,
        retain_graph=True,
    )[0]
    linear = torch.sum(reference_gradient * displacement)
    quadratic = 0.5 * torch.sum(displacement * hessian_times_u)
    remainder = raw_energy - reference_energy - linear - quadratic
    forces = -torch.autograd.grad(
        remainder.sum(),
        x,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    return TaylorResult(
        energy=remainder,
        forces_reference_order=forces,
        raw_energy=raw_energy,
        reference_energy=reference_energy,
        linear_term=linear.reshape(1),
        quadratic_term=quadratic.reshape(1),
        displacement=displacement,
        current_positions_reference_order=x,
    )


def taylor_remainder_energy_and_forces(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    current_positions_reference_order: torch.Tensor,
    reference_positions: torch.Tensor,
    image_integer_reference_order: torch.Tensor,
    *,
    create_graph: bool,
) -> TaylorResult:
    """Evaluate the complete scalar Taylor remainder and its autograd force."""
    if current_positions_reference_order.shape != reference_positions.shape:
        raise ValueError("current/reference position shapes differ")
    if image_integer_reference_order.shape != reference_positions.shape:
        raise ValueError("MIC image tensor shape differs")
    if image_integer_reference_order.requires_grad:
        raise ValueError("discrete MIC image integers must be detached")
    x = current_positions_reference_order
    cell = data["cell"].reshape(-1, 3, 3)[0]
    image_shift = image_integer_reference_order.to(dtype=x.dtype) @ cell
    return whole_energy_taylor2_remainder(
        lambda positions: mace_interaction_energy(model, data, positions),
        x,
        reference_positions,
        image_shift,
        create_graph=create_graph,
    )


def source_order_forces(
    forces_reference_order: torch.Tensor, assignment: Assignment
) -> torch.Tensor:
    result = torch.empty_like(forces_reference_order)
    indices = torch.as_tensor(
        assignment.reference_to_source,
        dtype=torch.long,
        device=forces_reference_order.device,
    )
    result[indices] = forces_reference_order
    return result


def evaluate_structure(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    *,
    device: torch.device | str,
    create_graph: bool,
    check_edges: bool = True,
) -> tuple[TaylorResult, Assignment]:
    """Order-safe public evaluation entry point for one complete structure."""
    validate_mace_architecture(model)
    reference = adapt_reference_cell(reference_template, structure)
    assignment = solve_assignment(structure, reference)
    ordered = reordered_structure(structure, assignment)
    # Align the ASE audit geometry with the same frozen MIC gauge used in Torch.
    aligned = ordered.copy()
    aligned.positions = (
        np.asarray(ordered.positions, float)
        - assignment.image_integer_reference_order @ np.asarray(ordered.cell, float)
    )
    if check_edges:
        validate_current_edge_set(aligned, reference)
    dtype = model_dtype(model)
    data = fixed_reference_graph(reference, device=device, dtype=dtype)
    positions = torch.as_tensor(
        np.asarray(ordered.positions, float), dtype=dtype, device=device
    ).clone().requires_grad_(True)
    reference_positions = torch.as_tensor(
        np.asarray(reference.positions, float), dtype=dtype, device=device
    )
    image = torch.as_tensor(
        assignment.image_integer_reference_order,
        dtype=torch.int64,
        device=device,
    )
    result = taylor_remainder_energy_and_forces(
        model,
        data,
        positions,
        reference_positions,
        image,
        create_graph=create_graph,
    )
    return result, assignment


def cutoff_c2_metrics(model: torch.nn.Module) -> dict[str, float]:
    """Audit the analytic inside-limit of the p=5 polynomial at r_max."""
    validate_mace_architecture(model)
    radius = torch.tensor([EXPECTED_R_MAX_A], dtype=torch.float64, requires_grad=True)
    p = torch.tensor(float(EXPECTED_CUTOFF_P), dtype=torch.float64)
    reduced = radius / EXPECTED_R_MAX_A
    # PolynomialCutoff's inside branch, intentionally without its x<rmax mask.
    value = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * reduced**p
        + p * (p + 2.0) * reduced ** (p + 1.0)
        - (p * (p + 1.0) / 2.0) * reduced ** (p + 2.0)
    )
    first = torch.autograd.grad(value.sum(), radius, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), radius, create_graph=True)[0]
    third = torch.autograd.grad(second.sum(), radius)[0]
    probes = {}
    parameter_device = next(model.parameters()).device
    for fractional_gap in (1.0e-2, 1.0e-3, 1.0e-4):
        probe = torch.tensor(
            [EXPECTED_R_MAX_A * (1.0 - fractional_gap)],
            dtype=model_dtype(model),
            device=parameter_device,
        )
        probes[f"fractional_gap_{fractional_gap:g}"] = float(
            model.radial_embedding.cutoff_fn(probe).detach()
        )
    return {
        "value": float(value.detach()),
        "first_derivative_A-1": float(first.detach()),
        "second_derivative_A-2": float(second.detach()),
        "third_derivative_A-3": float(third.detach()),
        "inside_value_probes": probes,
    }


def state_dict_sha256(model: torch.nn.Module) -> str:
    """Deterministic semantic hash of tensor names/dtypes/shapes/bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def finite_parameters_and_gradients(model: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter).all())
        and (parameter.grad is None or bool(torch.isfinite(parameter.grad).all()))
        for parameter in model.parameters()
    )
