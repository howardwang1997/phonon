#!/usr/bin/env python3
"""Conservative C2 routed-tail model for the support-free R2M core.

This module contains no training or model-selection policy.  It only defines
the fixed, differentiable correction

    E_tail = sum_i g_i(s_i(R)) * (epsilon_i(R) - c_C),

where the frozen R2M MACE core supplies equivariant node features, ``g`` is a
C2 smootherstep gate, and ``c_C`` is evaluated on a pristine 6x6 graphene
reference.  Forces are always obtained from the complete scalar energy.  In
particular, no ``g * F`` shortcut is provided by this API.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAT = "graphene_r2m_routed_tail_v1"
FEATURE_SCHEMA_VERSION = "graphene_r2m_core_invariants_v1"
EXPECTED_CORE_R_MAX_A = 2.0
EXPECTED_CORE_INTERACTIONS = 2
EXPECTED_RAW_NODE_WIDTH = 160
EXPECTED_INVARIANT_WIDTH = 64
EXPECTED_LAYOUT = (
    (0, 16, 0, 1),
    (0, 16, 1, -1),
    (0, 16, 2, 1),
    (1, 16, 0, 1),
)
EXPECTED_SCHEMA_TOP_LEVEL_KEYS = {
    "version",
    "core_r_max_A",
    "core_num_interactions",
    "raw_node_feats_width",
    "output_dimension",
    "blocks",
    "feature_names",
    "power_normalization",
    "accepted_parity",
}
EXPECTED_SCHEMA_BLOCK_KEYS = {
    "interaction",
    "multiplicity",
    "l",
    "parity",
    "dimension",
    "raw_slice",
    "operation",
    "output_slice",
    "feature_names",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def smootherstep(value: torch.Tensor) -> torch.Tensor:
    """Quintic step with zero first and second derivatives at 0 and 1."""
    clipped = torch.clamp(value, 0.0, 1.0)
    return clipped**3 * (10.0 - 15.0 * clipped + 6.0 * clipped**2)


def _core_layout(core: torch.nn.Module) -> tuple[list[dict], int]:
    if abs(float(core.r_max) - EXPECTED_CORE_R_MAX_A) > 1.0e-12:
        raise ValueError(
            f"routed tail requires r_max={EXPECTED_CORE_R_MAX_A:g} A; "
            f"observed {float(core.r_max):g} A"
        )
    if int(core.num_interactions) != EXPECTED_CORE_INTERACTIONS:
        raise ValueError(
            f"routed tail requires {EXPECTED_CORE_INTERACTIONS} interactions; "
            f"observed {int(core.num_interactions)}"
        )
    blocks: list[dict] = []
    offset = 0
    observed = []
    for interaction, product in enumerate(core.products):
        for multiplicity, irrep in product.linear.irreps_out:
            multiplicity = int(multiplicity)
            dimension = int(irrep.dim)
            angular_momentum = int(irrep.l)
            parity = int(irrep.p)
            width = multiplicity * dimension
            if angular_momentum == 0 and parity != 1:
                raise ValueError("0o pseudoscalars cannot enter the O(3) router")
            blocks.append(
                {
                    "interaction": int(interaction),
                    "multiplicity": multiplicity,
                    "l": angular_momentum,
                    "parity": parity,
                    "dimension": dimension,
                    "raw_slice": [offset, offset + width],
                    "operation": (
                        "signed_l0_value"
                        if angular_momentum == 0
                        else "mean_m_squared"
                    ),
                }
            )
            observed.append(
                (int(interaction), multiplicity, angular_momentum, parity)
            )
            offset += width
    if tuple(observed) != EXPECTED_LAYOUT:
        raise ValueError(
            "passing R2M core irreps layout changed: "
            f"expected {EXPECTED_LAYOUT}, observed {tuple(observed)}"
        )
    if offset != EXPECTED_RAW_NODE_WIDTH:
        raise ValueError(
            f"expected raw node width {EXPECTED_RAW_NODE_WIDTH}, observed {offset}"
        )
    return blocks, offset


def core_invariant_schema(core: torch.nn.Module) -> dict:
    """Freeze the exact 64-D invariant map of the passing R2M core."""
    blocks, raw_width = _core_layout(core)
    output_offset = 0
    names: list[str] = []
    for block in blocks:
        block_names = [
            (
                f"interaction{block['interaction']}_l{block['l']}_"
                f"p{block['parity']:+d}_channel{channel:02d}_"
                + (
                    "value"
                    if block["operation"] == "signed_l0_value"
                    else "mean_m_h2"
                )
            )
            for channel in range(block["multiplicity"])
        ]
        block["output_slice"] = [output_offset, output_offset + len(block_names)]
        block["feature_names"] = block_names
        names.extend(block_names)
        output_offset += len(block_names)
    if output_offset != EXPECTED_INVARIANT_WIDTH:
        raise ValueError(
            f"expected {EXPECTED_INVARIANT_WIDTH} invariants, observed {output_offset}"
        )
    payload = {
        "version": FEATURE_SCHEMA_VERSION,
        "core_r_max_A": EXPECTED_CORE_R_MAX_A,
        "core_num_interactions": EXPECTED_CORE_INTERACTIONS,
        "raw_node_feats_width": raw_width,
        "output_dimension": output_offset,
        "blocks": blocks,
        "feature_names": names,
        "power_normalization": "mean over m = squared norm/(2*l+1)",
        "accepted_parity": "signed 0e plus O(3)-even squared norms; reject 0o",
    }
    payload["schema_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_invariant_schema(schema: dict) -> None:
    payload = dict(schema)
    recorded_hash = payload.pop("schema_sha256", None)
    if recorded_hash is None or canonical_json_sha256(payload) != recorded_hash:
        raise ValueError("R2M invariant schema SHA-256 mismatch")
    if set(payload) != EXPECTED_SCHEMA_TOP_LEVEL_KEYS:
        raise ValueError("R2M invariant schema has unexpected top-level fields")
    if payload.get("version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported R2M invariant schema version")
    if float(payload.get("core_r_max_A", float("nan"))) != EXPECTED_CORE_R_MAX_A:
        raise ValueError("R2M invariant schema has the wrong core cutoff")
    if int(payload.get("core_num_interactions", -1)) != EXPECTED_CORE_INTERACTIONS:
        raise ValueError("R2M invariant schema has the wrong interaction count")
    if int(payload.get("raw_node_feats_width", -1)) != EXPECTED_RAW_NODE_WIDTH:
        raise ValueError("R2M raw node feature width changed")
    if int(payload.get("output_dimension", -1)) != EXPECTED_INVARIANT_WIDTH:
        raise ValueError("R2M invariant feature width changed")
    if payload.get("power_normalization") != (
        "mean over m = squared norm/(2*l+1)"
    ):
        raise ValueError("R2M invariant power normalization changed")
    if payload.get("accepted_parity") != (
        "signed 0e plus O(3)-even squared norms; reject 0o"
    ):
        raise ValueError("R2M invariant parity contract changed")
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != len(EXPECTED_LAYOUT):
        raise ValueError("R2M invariant schema has the wrong block count")
    expected_raw_start = 0
    expected_output_start = 0
    expected_all_names: list[str] = []
    for block, (interaction, multiplicity, angular_momentum, parity) in zip(
        blocks, EXPECTED_LAYOUT, strict=True
    ):
        if not isinstance(block, dict) or set(block) != EXPECTED_SCHEMA_BLOCK_KEYS:
            raise ValueError("R2M invariant schema block fields changed")
        dimension = 2 * angular_momentum + 1
        raw_stop = expected_raw_start + multiplicity * dimension
        output_stop = expected_output_start + multiplicity
        operation = (
            "signed_l0_value" if angular_momentum == 0 else "mean_m_squared"
        )
        expected_names = [
            (
                f"interaction{interaction}_l{angular_momentum}_"
                f"p{parity:+d}_channel{channel:02d}_"
                + ("value" if operation == "signed_l0_value" else "mean_m_h2")
            )
            for channel in range(multiplicity)
        ]
        expected = {
            "interaction": interaction,
            "multiplicity": multiplicity,
            "l": angular_momentum,
            "parity": parity,
            "dimension": dimension,
            "raw_slice": [expected_raw_start, raw_stop],
            "operation": operation,
            "output_slice": [expected_output_start, output_stop],
            "feature_names": expected_names,
        }
        if block != expected:
            raise ValueError("R2M invariant schema block mapping changed")
        expected_raw_start = raw_stop
        expected_output_start = output_stop
        expected_all_names.extend(expected_names)
    if payload.get("feature_names") != expected_all_names:
        raise ValueError("R2M invariant feature names/order changed")


def node_invariants(node_feats: torch.Tensor, schema: dict) -> torch.Tensor:
    """Map equivariant MACE node features to invariant scalars in Torch."""
    validate_invariant_schema(schema)
    if node_feats.ndim != 2 or node_feats.shape[1] != EXPECTED_RAW_NODE_WIDTH:
        raise ValueError(
            f"node_feats must have shape (n,{EXPECTED_RAW_NODE_WIDTH}); "
            f"observed {tuple(node_feats.shape)}"
        )
    values = []
    for item in schema["blocks"]:
        start, stop = map(int, item["raw_slice"])
        block = node_feats[:, start:stop].reshape(
            node_feats.shape[0], int(item["multiplicity"]), int(item["dimension"])
        )
        if item["operation"] == "signed_l0_value":
            values.append(block[:, :, 0])
        elif item["operation"] == "mean_m_squared":
            values.append(torch.mean(torch.square(block), dim=-1))
        else:
            raise ValueError(f"unknown invariant operation {item['operation']!r}")
    result = torch.cat(values, dim=1)
    if result.shape[1] != EXPECTED_INVARIANT_WIDTH:
        raise RuntimeError("R2M invariant replay produced the wrong width")
    return result


@dataclass(frozen=True)
class RoutedTailSpecification:
    input_dimension: int = EXPECTED_INVARIANT_WIDTH
    tail_hidden_1: int = 64
    tail_hidden_2: int = 32
    router_hidden_1: int = 32
    router_hidden_2: int = 16
    gate_score_off: float = 0.0
    gate_score_on: float = 1.0

    def validate(self) -> None:
        if self.input_dimension != EXPECTED_INVARIANT_WIDTH:
            raise ValueError("R2M routed-tail input dimension is frozen at 64")
        if min(
            self.tail_hidden_1,
            self.tail_hidden_2,
            self.router_hidden_1,
            self.router_hidden_2,
        ) <= 0:
            raise ValueError("hidden dimensions must be positive")
        if not self.gate_score_on > self.gate_score_off:
            raise ValueError("gate score interval is empty")


class RoutedTail(torch.nn.Module):
    """Independent tail/router MLPs on frozen, standardized core invariants."""

    def __init__(
        self,
        specification: RoutedTailSpecification,
        feature_mean: np.ndarray | torch.Tensor,
        feature_scale: np.ndarray | torch.Tensor,
        pristine_invariants: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        specification.validate()
        self.specification = specification
        mean = torch.as_tensor(feature_mean)
        scale = torch.as_tensor(feature_scale)
        pristine = torch.as_tensor(pristine_invariants)
        if mean.ndim != 1 or mean.shape != (specification.input_dimension,):
            raise ValueError("feature mean has the wrong shape")
        if (
            scale.shape != mean.shape
            or not torch.all(torch.isfinite(mean))
            or not torch.all(torch.isfinite(scale))
            or torch.any(scale <= 0.0)
        ):
            raise ValueError("feature scales must be positive and match the mean")
        if pristine.ndim != 2 or pristine.shape[1] != specification.input_dimension:
            raise ValueError("pristine invariants have the wrong shape")
        if not torch.all(torch.isfinite(pristine)):
            raise ValueError("pristine invariants must be finite")
        dtype = torch.promote_types(mean.dtype, pristine.dtype)
        if not dtype.is_floating_point:
            dtype = torch.float32
        self.register_buffer("feature_mean", mean.to(dtype=dtype).clone())
        self.register_buffer("feature_scale", scale.to(dtype=dtype).clone())
        self.register_buffer("pristine_invariants", pristine.to(dtype=dtype).clone())
        self.tail_encoder = torch.nn.Sequential(
            torch.nn.Linear(specification.input_dimension, specification.tail_hidden_1),
            torch.nn.SiLU(),
            torch.nn.Linear(specification.tail_hidden_1, specification.tail_hidden_2),
            torch.nn.SiLU(),
            torch.nn.Linear(specification.tail_hidden_2, 1),
        ).to(dtype=dtype)
        self.router_encoder = torch.nn.Sequential(
            torch.nn.Linear(
                specification.input_dimension, specification.router_hidden_1
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                specification.router_hidden_1, specification.router_hidden_2
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(specification.router_hidden_2, 1),
        ).to(dtype=dtype)
        torch.nn.init.zeros_(self.tail_encoder[-1].weight)
        torch.nn.init.zeros_(self.tail_encoder[-1].bias)
        torch.nn.init.zeros_(self.router_encoder[-1].weight)
        torch.nn.init.constant_(self.router_encoder[-1].bias, 0.5)

    @property
    def dtype(self) -> torch.dtype:
        return self.feature_mean.dtype

    def standardized(self, invariants: torch.Tensor) -> torch.Tensor:
        if invariants.ndim != 2 or invariants.shape[1] != self.specification.input_dimension:
            raise ValueError("routed-tail invariants have the wrong shape")
        return (
            invariants.to(dtype=self.dtype) - self.feature_mean[None, :]
        ) / self.feature_scale[None, :]

    def pristine_carbon_gauge(self) -> torch.Tensor:
        standardized = self.standardized(self.pristine_invariants)
        return torch.mean(self.tail_encoder(standardized).squeeze(-1))

    def node_terms(
        self, invariants: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        standardized = self.standardized(invariants)
        epsilon = self.tail_encoder(standardized).squeeze(-1)
        score = self.router_encoder(standardized).squeeze(-1)
        interval = self.specification.gate_score_on - self.specification.gate_score_off
        gate = smootherstep(
            (score - self.specification.gate_score_off) / interval
        )
        c_c = self.pristine_carbon_gauge()
        node_energy = gate * (epsilon - c_c)
        return node_energy, gate, score, epsilon

    def graph_energies(
        self, invariants: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if batch.ndim != 1 or batch.shape[0] != invariants.shape[0]:
            raise ValueError("batch vector does not match node invariants")
        if batch.dtype != torch.long:
            batch = batch.to(dtype=torch.long)
        node_energy, gate, score, epsilon = self.node_terms(invariants)
        n_graphs = int(batch.max().detach().cpu()) + 1 if batch.numel() else 0
        energy = node_energy.new_zeros(n_graphs)
        energy.index_add_(0, batch, node_energy)
        return energy, {
            "node_energy": node_energy,
            "gate": gate,
            "score": score,
            "epsilon": epsilon,
            "c_C_eV": self.pristine_carbon_gauge(),
        }


def ensure_frozen_core(core: torch.nn.Module) -> None:
    trainable = [name for name, parameter in core.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError(
            "R2M core must be frozen before routed-tail autograd; trainable: "
            + ", ".join(trainable[:5])
        )


def correction_energy_and_forces(
    core: torch.nn.Module,
    tail: RoutedTail,
    data: dict[str, torch.Tensor],
    feature_schema: dict,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate the complete conservative routed energy and its force.

    The only force path is ``-grad(sum(E_tail), positions)``.  Core parameters
    are frozen, while their node-feature dependence on query positions remains
    live.  This includes both ``-g grad(epsilon)`` and ``-epsilon grad(g)`` as
    well as every cross-atom contribution of the MACE encoder.
    """
    ensure_frozen_core(core)
    validate_invariant_schema(feature_schema)
    model_data = dict(data)
    positions = model_data["positions"]
    if not positions.requires_grad:
        positions = positions.detach().clone().requires_grad_(True)
    model_data["positions"] = positions
    output = core(model_data, training=True, compute_force=False)
    invariants = node_invariants(output["node_feats"], feature_schema)
    energies, diagnostics = tail.graph_energies(invariants, model_data["batch"])
    forces = -torch.autograd.grad(
        torch.sum(energies),
        positions,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    diagnostics.update(
        {
            "core_energy": output.get("energy"),
            "invariants": invariants,
            "positions": positions,
        }
    )
    return energies, forces, diagnostics


def save_routed_tail(
    path: Path,
    model: RoutedTail,
    feature_schema: dict,
    core_sha256: str,
    metadata: dict | None = None,
) -> None:
    validate_invariant_schema(feature_schema)
    if len(core_sha256) != 64:
        raise ValueError("core SHA-256 must be a 64-character hex digest")
    int(core_sha256, 16)
    payload = {
        "format": FORMAT,
        "specification": asdict(model.specification),
        "feature_schema": feature_schema,
        "core_sha256": core_sha256,
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "metadata": dict(metadata or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_routed_tail(
    path: Path,
    device: torch.device,
    expected_core_sha256: str,
) -> tuple[RoutedTail, dict, dict]:
    payload = torch_load(path, map_location="cpu")
    if payload.get("format") != FORMAT:
        raise ValueError("unsupported R2M routed-tail artifact")
    if payload.get("core_sha256") != expected_core_sha256:
        raise ValueError("routed tail was trained against another core")
    schema = payload["feature_schema"]
    validate_invariant_schema(schema)
    specification = RoutedTailSpecification(**payload["specification"])
    state = payload["state_dict"]
    model = RoutedTail(
        specification,
        state["feature_mean"],
        state["feature_scale"],
        state["pristine_invariants"],
    )
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    return model, schema, dict(payload.get("metadata", {}))
