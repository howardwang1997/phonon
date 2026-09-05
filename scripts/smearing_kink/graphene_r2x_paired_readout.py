#!/usr/bin/env python3
"""Runtime energy/force/Hessian API for the frozen R2X paired readout."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

import graphene_r2r_multipolar_background as r2r


FORMAT = "graphene_r2x_paired_bilinear_runtime_v1"
SELECTED_BILINEAR_INDICES = np.asarray(
    [
        0,
        17,
        34,
        51,
        68,
        85,
        102,
        119,
        136,
        153,
        170,
        187,
        204,
        221,
        238,
        255,
        256,
        273,
        290,
        307,
        324,
        341,
        358,
        375,
        392,
        409,
        426,
        443,
        460,
        477,
        494,
        511,
        127,
        383,
    ],
    dtype=np.int64,
)
CHECKPOINT_MEMBERS = {
    "physical_coefficient",
    "base65_physical_coefficient",
    "bilinear34_physical_coefficient",
    "selected_bilinear_indices",
    "feature_mean",
    "feature_scale",
    "OOF_predicted_force_eV_A",
    "train_predicted_force_eV_A",
    "thermal_energy_design_eV",
    "thermal_force_design_eV_A",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype=np.dtype("<f8"), order="C")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _frozen_float64(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.str != "<f8" or not array.flags.c_contiguous:
        raise ValueError(f"R2X {label} schema changed")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"R2X {label} contains a non-finite value")
    output = np.array(array, dtype="<f8", order="C", copy=True)
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class PairedReadoutCheckpoint:
    path: Path
    file_sha256: str
    physical_coefficient: np.ndarray = field(repr=False)
    feature_mean: np.ndarray = field(repr=False)
    feature_scale: np.ndarray = field(repr=False)
    selected_bilinear_indices: np.ndarray = field(repr=False)
    coefficient_raw_sha256: str

    def validate(self) -> None:
        if file_sha256(self.path) != self.file_sha256:
            raise ValueError("R2X frozen readout file changed after loading")
        if self.physical_coefficient.flags.writeable:
            raise ValueError("R2X in-memory coefficient became writable")
        if raw_array_sha256(self.physical_coefficient) != self.coefficient_raw_sha256:
            raise ValueError("R2X in-memory coefficient changed after loading")


def load_paired_readout_checkpoint(
    path: Path, *, expected_sha256: str
) -> PairedReadoutCheckpoint:
    source = Path(path).resolve(strict=True)
    observed = file_sha256(source)
    if observed != expected_sha256:
        raise ValueError("R2X readout NPZ differs from the externally supplied SHA256")
    with np.load(source, allow_pickle=False) as arrays:
        if set(arrays.files) != CHECKPOINT_MEMBERS:
            raise ValueError("R2X readout NPZ member set changed")
        coefficient = _frozen_float64(
            arrays["physical_coefficient"], (99,), "physical coefficient"
        )
        base = _frozen_float64(
            arrays["base65_physical_coefficient"], (65,), "base65 coefficient"
        )
        bilinear = _frozen_float64(
            arrays["bilinear34_physical_coefficient"],
            (34,),
            "bilinear34 coefficient",
        )
        mean = _frozen_float64(arrays["feature_mean"], (34,), "feature mean")
        scale = _frozen_float64(arrays["feature_scale"], (34,), "feature scale")
        indices = np.asarray(arrays["selected_bilinear_indices"])
    if not np.array_equal(coefficient[:65], base) or not np.array_equal(
        coefficient[65:], bilinear
    ):
        raise ValueError("R2X split coefficient arrays do not replay physical coefficient")
    if indices.shape != (34,) or indices.dtype.str != "<i8" or not np.array_equal(
        indices, SELECTED_BILINEAR_INDICES
    ):
        raise ValueError("R2X selected bilinear ordering changed")
    if np.any(scale <= 0.0):
        raise ValueError("R2X feature scale must be positive")
    frozen_indices = np.array(indices, dtype="<i8", order="C", copy=True)
    frozen_indices.setflags(write=False)
    return PairedReadoutCheckpoint(
        path=source,
        file_sha256=observed,
        physical_coefficient=coefficient,
        feature_mean=mean,
        feature_scale=scale,
        selected_bilinear_indices=frozen_indices,
        coefficient_raw_sha256=raw_array_sha256(coefficient),
    )


def selected_bilinear_energy_columns(
    normalized_features: torch.Tensor,
    multipolar_gate: torch.Tensor,
    amplitude_gate: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    if normalized_features.ndim != 2 or normalized_features.shape[1] != 34:
        raise ValueError("R2X normalized node feature shape changed")
    first = normalized_features[:, :16]
    second = normalized_features[:, 16:32]
    plain = torch.einsum(
        "n,nj,nk->jk", multipolar_gate, first, second
    ).reshape(-1)
    modulated = torch.einsum(
        "n,n,nj,nk->jk",
        multipolar_gate,
        amplitude_gate,
        first,
        second,
    ).reshape(-1)
    full = torch.cat((plain, modulated))
    if full.shape != (512,):
        raise ValueError("R2X full bilinear scalar layout changed")
    return full[selected_indices]


def _combined_scalar_context(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    checkpoint: PairedReadoutCheckpoint,
    *,
    device: torch.device | str,
    formal_graph_data: Mapping[str, torch.Tensor] | None = None,
    graph_mode: str = "baseline",
    baseline_reference_template: Atoms | None = None,
    baseline_structure_template: Atoms | None = None,
    rigid_transform: np.ndarray | None = None,
) -> tuple[Any, torch.Tensor]:
    checkpoint.validate()
    context = r2r._verified_production_context(
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
    if context.current_positions_reference_order.dtype != torch.float64:
        raise ValueError("R2X runtime requires FP64 production positions")
    fixed, base_columns = r2r._production_energy_components(context)
    live = context.current_positions_reference_order
    aligned = live - context.image_shift.detach()
    observables = context.node_observables_fn(aligned)
    fields = context.background_fields_fn(aligned)
    node_features = torch.cat(
        (
            observables.signed_l0,
            observables.scale_shift_node_energy_eV[:, None],
            fields.amplitude_gate[:, None],
        ),
        dim=1,
    )
    mean = torch.as_tensor(
        np.array(checkpoint.feature_mean, copy=True),
        dtype=torch.float64,
        device=live.device,
    )
    scale = torch.as_tensor(
        np.array(checkpoint.feature_scale, copy=True),
        dtype=torch.float64,
        device=live.device,
    )
    normalized = (node_features - mean) / scale
    selected = torch.as_tensor(
        np.array(checkpoint.selected_bilinear_indices, copy=True),
        dtype=torch.long,
        device=live.device,
    )
    bilinear_columns = selected_bilinear_energy_columns(
        normalized,
        fields.multipolar_gate,
        fields.amplitude_gate,
        selected,
    )
    coefficient = torch.as_tensor(
        np.array(checkpoint.physical_coefficient, copy=True),
        dtype=torch.float64,
        device=live.device,
    )
    energy = (
        fixed
        + torch.dot(base_columns, coefficient[:65])
        + torch.dot(bilinear_columns, coefficient[65:])
    )
    if energy.ndim != 0 or not bool(torch.isfinite(energy)):
        raise ValueError("R2X combined scalar is invalid")
    return context, energy


@dataclass
class PairedReadoutProbe:
    energy_eV: torch.Tensor
    force_source_order_eV_A: torch.Tensor
    coefficient_raw_sha256: str


@dataclass
class PairedReadoutMechanics(PairedReadoutProbe):
    Hessian_source_order_eV_A2: torch.Tensor
    Hessian_antisymmetry_max_abs_eV_A2: float
    Hessian_translation_ASR_max_abs_eV_A2: float


def production_paired_readout_energy_force(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    checkpoint: PairedReadoutCheckpoint,
    *,
    device: torch.device | str,
    formal_graph_data: Mapping[str, torch.Tensor] | None = None,
    graph_mode: str = "baseline",
    baseline_reference_template: Atoms | None = None,
    baseline_structure_template: Atoms | None = None,
    rigid_transform: np.ndarray | None = None,
) -> PairedReadoutProbe:
    context, energy = _combined_scalar_context(
        model,
        structure,
        reference_template,
        checkpoint,
        device=device,
        formal_graph_data=formal_graph_data,
        graph_mode=graph_mode,
        baseline_reference_template=baseline_reference_template,
        baseline_structure_template=baseline_structure_template,
        rigid_transform=rigid_transform,
    )
    force_reference = -torch.autograd.grad(
        energy, context.current_positions_reference_order, create_graph=False
    )[0]
    force_source = r2r._source_order_force(force_reference, context.assignment)
    if not bool(torch.isfinite(force_source).all()):
        raise ValueError("R2X source-order force is invalid")
    return PairedReadoutProbe(
        energy_eV=energy,
        force_source_order_eV_A=force_source,
        coefficient_raw_sha256=checkpoint.coefficient_raw_sha256,
    )


def production_paired_readout_energy_force_hessian(
    model: torch.nn.Module,
    structure: Atoms,
    reference_template: Atoms,
    checkpoint: PairedReadoutCheckpoint,
    *,
    device: torch.device | str,
    formal_graph_data: Mapping[str, torch.Tensor] | None = None,
    graph_mode: str = "baseline",
    baseline_reference_template: Atoms | None = None,
    baseline_structure_template: Atoms | None = None,
    rigid_transform: np.ndarray | None = None,
) -> PairedReadoutMechanics:
    context, energy = _combined_scalar_context(
        model,
        structure,
        reference_template,
        checkpoint,
        device=device,
        formal_graph_data=formal_graph_data,
        graph_mode=graph_mode,
        baseline_reference_template=baseline_reference_template,
        baseline_structure_template=baseline_structure_template,
        rigid_transform=rigid_transform,
    )
    live = context.current_positions_reference_order
    force_reference = -torch.autograd.grad(
        energy, live, create_graph=True, retain_graph=True
    )[0]
    rows = []
    for component in force_reference.reshape(-1):
        derivative = torch.autograd.grad(component, live, retain_graph=True)[0]
        rows.append(-derivative.reshape(-1))
    reference_hessian = torch.stack(rows)
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
    force_source = r2r._source_order_force(force_reference, context.assignment)
    if not bool(torch.isfinite(force_source).all()) or not bool(
        torch.isfinite(hessian).all()
    ):
        raise ValueError("R2X full mechanics contains a non-finite value")
    atom_count = context.atom_count
    blocks = hessian.reshape(atom_count, 3, atom_count, 3)
    antisymmetry = float(torch.max(torch.abs(hessian - hessian.T)).detach().cpu())
    translation_asr = float(torch.max(torch.abs(blocks.sum(dim=2))).detach().cpu())
    if not math.isfinite(antisymmetry) or not math.isfinite(translation_asr):
        raise ValueError("R2X Hessian diagnostics are non-finite")
    return PairedReadoutMechanics(
        energy_eV=energy,
        force_source_order_eV_A=force_source,
        coefficient_raw_sha256=checkpoint.coefficient_raw_sha256,
        Hessian_source_order_eV_A2=hessian,
        Hessian_antisymmetry_max_abs_eV_A2=antisymmetry,
        Hessian_translation_ASR_max_abs_eV_A2=translation_asr,
    )


class R2XCorrectionCalculator(Calculator):
    """ASE calculator exposing only the frozen anharmonic R2X correction."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model: torch.nn.Module,
        reference_template: Atoms,
        checkpoint: PairedReadoutCheckpoint,
        *,
        device: torch.device | str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.reference_template = reference_template.copy()
        self.checkpoint = checkpoint
        self.device = device

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if self.atoms is None:
            raise ValueError("R2X ASE calculator requires atoms")
        with torch.enable_grad():
            probe = production_paired_readout_energy_force(
                self.model,
                self.atoms,
                self.reference_template,
                self.checkpoint,
                device=self.device,
            )
        self.results = {
            "energy": float(probe.energy_eV.detach().cpu()),
            "forces": probe.force_source_order_eV_A.detach().cpu().numpy(),
        }


__all__ = [
    "FORMAT",
    "PairedReadoutCheckpoint",
    "PairedReadoutMechanics",
    "PairedReadoutProbe",
    "R2XCorrectionCalculator",
    "SELECTED_BILINEAR_INDICES",
    "load_paired_readout_checkpoint",
    "production_paired_readout_energy_force",
    "production_paired_readout_energy_force_hessian",
    "selected_bilinear_energy_columns",
]
