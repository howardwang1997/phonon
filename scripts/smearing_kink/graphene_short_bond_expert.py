#!/usr/bin/env python3
"""Compact, conservative short-bond expert for graphene.

The expert adds a scalar energy only for compressed first-neighbour C--C
bonds.  A C2 switching function makes both the energy contribution and its
first two radial derivatives vanish at the activation boundaries.  The
nonlinear variant conditions a high-resolution radial expansion on symmetric
endpoint environments; the constant variant is the corresponding radial-only
control.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


FORMAT = "graphene_short_bond_expert_v1"


@dataclass(frozen=True)
class ExpertSpecification:
    neighbour_cutoff_A: float = 1.75
    lower_off_A: float = 1.15
    lower_on_A: float = 1.20
    upper_on_A: float = 1.32
    upper_off_A: float = 1.38
    radial_min_A: float = 1.20
    radial_max_A: float = 1.36
    n_radial: int = 12
    radial_width_A: float = 0.022
    hidden_channels: int = 24
    environment_mode: str = "nonlinear"

    def validate(self) -> None:
        if not (
            self.lower_off_A
            < self.lower_on_A
            < self.upper_on_A
            < self.upper_off_A
            < self.neighbour_cutoff_A
        ):
            raise ValueError("invalid compact activation interval")
        if not (
            self.lower_on_A <= self.radial_min_A < self.radial_max_A <= self.upper_off_A
        ):
            raise ValueError("radial centers must lie inside the active interval")
        if self.n_radial < 4 or self.radial_width_A <= 0.0:
            raise ValueError("invalid radial resolution")
        if self.environment_mode not in {"nonlinear", "constant"}:
            raise ValueError("environment_mode must be nonlinear or constant")
        if self.environment_mode == "nonlinear" and self.hidden_channels <= 0:
            raise ValueError("nonlinear expert requires positive hidden_channels")


@dataclass(frozen=True)
class Topology:
    left: np.ndarray
    right: np.ndarray
    shift_cart_A: np.ndarray
    other_left: np.ndarray
    other_left_sign: np.ndarray
    other_right: np.ndarray
    other_right_sign: np.ndarray


def build_topology(structure, specification: ExpertSpecification) -> Topology:
    """Build a fixed three-fold first-neighbour topology for one structure."""
    positions = np.asarray(structure.positions, float)
    cell = np.asarray(structure.cell, float)
    delta = positions[None, :, :] - positions[:, None, :]
    fractional = delta @ np.linalg.inv(cell)
    image_shift = -np.round(fractional)
    vectors = delta + image_shift @ cell
    distances = np.linalg.norm(vectors, axis=2)
    left, right = np.where(
        np.triu(
            (distances > 1.0e-8)
            & (distances < specification.neighbour_cutoff_A),
            k=1,
        )
    )
    expected = 3 * len(structure) // 2
    if len(left) != expected:
        raise ValueError(
            f"expected {expected} first-neighbour bonds, found {len(left)}"
        )
    shift_cart = image_shift[left, right] @ cell

    incident: list[list[tuple[int, float]]] = [[] for _ in structure]
    for bond, (atom_i, atom_j) in enumerate(zip(left, right, strict=True)):
        incident[int(atom_i)].append((bond, 1.0))
        incident[int(atom_j)].append((bond, -1.0))
    if any(len(items) != 3 for items in incident):
        raise ValueError("first-shell coordination is not exactly three")

    other_left = []
    other_left_sign = []
    other_right = []
    other_right_sign = []
    for bond, (atom_i, atom_j) in enumerate(zip(left, right, strict=True)):
        left_items = [item for item in incident[int(atom_i)] if item[0] != bond]
        right_items = [item for item in incident[int(atom_j)] if item[0] != bond]
        other_left.append([item[0] for item in left_items])
        other_left_sign.append([item[1] for item in left_items])
        other_right.append([item[0] for item in right_items])
        other_right_sign.append([item[1] for item in right_items])
    return Topology(
        left=np.asarray(left, dtype=np.int64),
        right=np.asarray(right, dtype=np.int64),
        shift_cart_A=np.asarray(shift_cart, float),
        other_left=np.asarray(other_left, dtype=np.int64),
        other_left_sign=np.asarray(other_left_sign, float),
        other_right=np.asarray(other_right, dtype=np.int64),
        other_right_sign=np.asarray(other_right_sign, float),
    )


def torch_topology(
    topology: Topology, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    return {
        "left": torch.as_tensor(topology.left, dtype=torch.long, device=device),
        "right": torch.as_tensor(topology.right, dtype=torch.long, device=device),
        "shift": torch.as_tensor(topology.shift_cart_A, dtype=dtype, device=device),
        "other_left": torch.as_tensor(
            topology.other_left, dtype=torch.long, device=device
        ),
        "other_left_sign": torch.as_tensor(
            topology.other_left_sign, dtype=dtype, device=device
        ),
        "other_right": torch.as_tensor(
            topology.other_right, dtype=torch.long, device=device
        ),
        "other_right_sign": torch.as_tensor(
            topology.other_right_sign, dtype=dtype, device=device
        ),
    }


def bond_geometry(
    positions: torch.Tensor, topology: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vectors = (
        positions[topology["right"]]
        - positions[topology["left"]]
        + topology["shift"]
    )
    distance = torch.linalg.vector_norm(vectors, dim=1)
    left_other_vectors = (
        vectors[topology["other_left"]]
        * topology["other_left_sign"][..., None]
    )
    right_other_vectors = (
        vectors[topology["other_right"]]
        * topology["other_right_sign"][..., None]
    )
    return distance, left_other_vectors, right_other_vectors


def raw_environment_features(
    positions: torch.Tensor, topology: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return central distance and endpoint/permutation-symmetric scalars."""
    distance, left_vectors, right_vectors = bond_geometry(positions, topology)
    left_distance = torch.linalg.vector_norm(left_vectors, dim=2)
    right_distance = torch.linalg.vector_norm(right_vectors, dim=2)
    central_vectors = (
        positions[topology["right"]]
        - positions[topology["left"]]
        + topology["shift"]
    )
    left_cosine = torch.sum(central_vectors[:, None, :] * left_vectors, dim=2) / (
        distance[:, None] * left_distance
    )
    right_cosine = torch.sum((-central_vectors)[:, None, :] * right_vectors, dim=2) / (
        distance[:, None] * right_distance
    )

    left_mean_distance = torch.mean(left_distance, dim=1)
    right_mean_distance = torch.mean(right_distance, dim=1)
    all_distance = torch.cat([left_distance, right_distance], dim=1)
    mean_distance = torch.mean(all_distance, dim=1)
    distance_variance = torch.mean(
        (all_distance - mean_distance[:, None]).square(), dim=1
    )
    endpoint_distance_asymmetry = (
        left_mean_distance - right_mean_distance
    ).square()
    within_endpoint_splitting = 0.5 * (
        (left_distance[:, 0] - left_distance[:, 1]).square()
        + (right_distance[:, 0] - right_distance[:, 1]).square()
    )

    angle_deviation = torch.cat([left_cosine + 0.5, right_cosine + 0.5], dim=1)
    mean_angle_deviation = torch.mean(angle_deviation, dim=1)
    mean_square_angle_deviation = torch.mean(angle_deviation.square(), dim=1)
    endpoint_angle_asymmetry = (
        torch.mean(left_cosine, dim=1) - torch.mean(right_cosine, dim=1)
    ).square()

    left_neighbour_cosine = torch.sum(
        left_vectors[:, 0, :] * left_vectors[:, 1, :], dim=1
    ) / (left_distance[:, 0] * left_distance[:, 1])
    right_neighbour_cosine = torch.sum(
        right_vectors[:, 0, :] * right_vectors[:, 1, :], dim=1
    ) / (right_distance[:, 0] * right_distance[:, 1])
    neighbour_angle_mean = 0.5 * (
        left_neighbour_cosine + right_neighbour_cosine
    ) + 0.5
    neighbour_angle_asymmetry = (
        left_neighbour_cosine - right_neighbour_cosine
    ).square()

    left_cosine_centered = left_cosine - torch.mean(left_cosine, dim=1)[:, None]
    right_cosine_centered = right_cosine - torch.mean(right_cosine, dim=1)[:, None]
    length_angle_covariance = 0.5 * (
        torch.mean(
            (left_distance - left_mean_distance[:, None]) * left_cosine_centered,
            dim=1,
        )
        + torch.mean(
            (right_distance - right_mean_distance[:, None])
            * right_cosine_centered,
            dim=1,
        )
    )

    raw = torch.stack(
        [
            mean_distance,
            mean_distance - distance,
            distance_variance,
            endpoint_distance_asymmetry,
            within_endpoint_splitting,
            mean_angle_deviation,
            mean_square_angle_deviation,
            endpoint_angle_asymmetry,
            neighbour_angle_mean,
            neighbour_angle_asymmetry,
            length_angle_covariance,
        ],
        dim=1,
    )
    return distance, raw


def smootherstep(value: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(value, 0.0, 1.0)
    return clipped**3 * (10.0 - 15.0 * clipped + 6.0 * clipped**2)


def activation_envelope(
    distance: torch.Tensor, specification: ExpertSpecification
) -> torch.Tensor:
    lower = smootherstep(
        (distance - specification.lower_off_A)
        / (specification.lower_on_A - specification.lower_off_A)
    )
    upper = 1.0 - smootherstep(
        (distance - specification.upper_on_A)
        / (specification.upper_off_A - specification.upper_on_A)
    )
    return lower * upper


class ShortBondExpert(torch.nn.Module):
    n_environment_features = 11

    def __init__(
        self,
        specification: ExpertSpecification,
        environment_mean: np.ndarray | torch.Tensor,
        environment_scale: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        specification.validate()
        self.specification = specification
        mean = torch.as_tensor(environment_mean, dtype=torch.float64)
        scale = torch.as_tensor(environment_scale, dtype=torch.float64)
        if mean.shape != (self.n_environment_features,) or scale.shape != mean.shape:
            raise ValueError("unexpected environment statistics shape")
        if torch.any(scale <= 0.0):
            raise ValueError("environment scales must be positive")
        self.register_buffer("environment_mean", mean.clone())
        self.register_buffer("environment_scale", scale.clone())
        self.register_buffer(
            "radial_centers_A",
            torch.linspace(
                specification.radial_min_A,
                specification.radial_max_A,
                specification.n_radial,
                dtype=torch.float64,
            ),
        )
        if specification.environment_mode == "nonlinear":
            hidden = specification.hidden_channels
            self.environment_network = torch.nn.Sequential(
                torch.nn.Linear(self.n_environment_features, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, specification.n_radial),
            ).to(dtype=torch.float64)
            torch.nn.init.zeros_(self.environment_network[-1].weight)
            torch.nn.init.zeros_(self.environment_network[-1].bias)
            self.radial_coefficients = None
        else:
            self.environment_network = None
            self.radial_coefficients = torch.nn.Parameter(
                torch.zeros(specification.n_radial, dtype=torch.float64)
            )

    def energy(
        self, positions: torch.Tensor, topology: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        distance, raw = raw_environment_features(positions, topology)
        envelope = activation_envelope(distance, self.specification)
        radial = torch.exp(
            -0.5
            * (
                (distance[:, None] - self.radial_centers_A[None, :])
                / self.specification.radial_width_A
            ).square()
        )
        if self.environment_network is None:
            coefficients = self.radial_coefficients[None, :].expand_as(radial)
        else:
            standardized = (
                raw - self.environment_mean[None, :]
            ) / self.environment_scale[None, :]
            coefficients = self.environment_network(standardized)
        bond_energy = envelope * torch.sum(radial * coefficients, dim=1)
        return torch.sum(bond_energy)

    def energy_and_forces(
        self,
        positions: torch.Tensor,
        topology: dict[str, torch.Tensor],
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)
        energy = self.energy(positions, topology)
        forces = -torch.autograd.grad(
            energy, positions, create_graph=create_graph, retain_graph=create_graph
        )[0]
        return energy, forces


def environment_statistics(
    structures: list,
    specification: ExpertSpecification,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    device = torch.device("cpu")
    with torch.no_grad():
        for structure in structures:
            topology = torch_topology(
                build_topology(structure, specification), device, torch.float64
            )
            positions = torch.as_tensor(
                np.asarray(structure.positions, float), dtype=torch.float64
            )
            _, raw = raw_environment_features(positions, topology)
            distance, _, _ = bond_geometry(positions, topology)
            active = activation_envelope(distance, specification) > 1.0e-8
            if torch.any(active):
                values.append(raw[active].cpu().numpy())
    if not values:
        raise ValueError("no bonds lie inside the short-bond activation interval")
    joined = np.concatenate(values, axis=0)
    mean = np.mean(joined, axis=0)
    scale = np.std(joined, axis=0)
    scale = np.maximum(scale, np.maximum(np.abs(mean) * 1.0e-6, 1.0e-8))
    return mean, scale


def expert_prediction(
    model: ShortBondExpert,
    structure,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    topology = torch_topology(
        build_topology(structure, model.specification), device, torch.float64
    )
    positions = torch.tensor(
        np.asarray(structure.positions, float),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    energy, forces = model.energy_and_forces(positions, topology, create_graph=False)
    return float(energy.detach().cpu()), forces.detach().cpu().numpy()


def save_expert(
    path: Path,
    model: ShortBondExpert,
    metadata: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "format": FORMAT,
        "specification": asdict(model.specification),
        "environment_mean": model.environment_mean.detach().cpu(),
        "environment_scale": model.environment_scale.detach().cpu(),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def load_expert(path: Path, device: torch.device) -> tuple[ShortBondExpert, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported expert format in {path}")
    specification = ExpertSpecification(**payload["specification"])
    model = ShortBondExpert(
        specification,
        payload["environment_mean"],
        payload["environment_scale"],
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device=device, dtype=torch.float64)
    return model, dict(payload.get("metadata", {}))
