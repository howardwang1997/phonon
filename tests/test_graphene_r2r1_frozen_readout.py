from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "smearing_kink"
sys.path.insert(0, str(SCRIPTS))

import graphene_r2o_taylor_null as r2o  # noqa: E402
import graphene_r2r0_formal as formal  # noqa: E402
import graphene_r2r1_frozen_readout as frozen  # noqa: E402
import graphene_r2r_multipolar_background as r2r  # noqa: E402


DATA = ROOT / "data" / "graphene_r2o_taylor_null_core"
RUN = (
    ROOT
    / "results"
    / "graphene_physics_temperature"
    / "post_p4_feasibility"
    / "R2Q_four_step_trust_region"
    / "formal_4step_seed83_rtx"
)
FIT_MANIFEST_SHA256 = "1" * 64
FIT_RECEIPT_SHA256 = "2" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(root: Path, physical_p: np.ndarray) -> tuple[Path, str]:
    root.mkdir()
    arrays_path = root / frozen.CHECKPOINT_ARRAY_BASENAME
    np.savez(arrays_path, physical_p=physical_p)
    receipt = frozen.checkpoint_receipt_payload(
        np.asarray(physical_p),
        array_sha256=_sha256(arrays_path),
        fit_manifest_sha256=FIT_MANIFEST_SHA256,
        fit_receipt_sha256=FIT_RECEIPT_SHA256,
    )
    receipt_path = root / frozen.CHECKPOINT_RECEIPT_BASENAME
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    receipt_sha = _sha256(receipt_path)
    (root / frozen.CHECKPOINT_MARKER_BASENAME).write_bytes(
        (frozen.CHECKPOINT_STATUS + "\n" + receipt_sha + "\n").encode("ascii")
    )
    return root, receipt_sha


def _rewrite_receipt(
    root: Path, mutation,
) -> str:
    path = root / frozen.CHECKPOINT_RECEIPT_BASENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    receipt_sha = _sha256(path)
    (root / frozen.CHECKPOINT_MARKER_BASENAME).write_bytes(
        (frozen.CHECKPOINT_STATUS + "\n" + receipt_sha + "\n").encode("ascii")
    )
    return receipt_sha


def _load(root: Path, receipt_sha: str) -> frozen.FrozenReadoutCheckpoint:
    return frozen.load_frozen_readout_checkpoint(
        root, expected_receipt_sha256=receipt_sha
    )


def _valid_context_receipt(atom_count: int) -> dict:
    return {
        "format": "synthetic_hash_bound_context",
        "endpoint_state_sha256": frozen.FROZEN_ENDPOINT_STATE_SHA256,
        "reference_semantic_sha256": "3" * 64,
        "formal_R2O_graph_semantic_sha256": "4" * 64,
        "assignment_and_MIC_semantic_sha256": "5" * 64,
        "affine_component_names_sha256": (
            frozen.FROZEN_AFFINE_COMPONENT_NAMES_SHA256
        ),
        "parameter_columns": 65,
        "fixed_offset_columns": 1,
        "source_order_force_rows": 3 * atom_count,
        "production_integration_API_only": True,
    }


def _install_synthetic_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def verified_context(
        model,
        structure,
        reference_template,
        *,
        device,
        formal_graph_data=None,
        graph_mode="baseline",
        baseline_reference_template=None,
        baseline_structure_template=None,
        rigid_transform=None,
    ):
        del (
            model,
            reference_template,
            formal_graph_data,
            graph_mode,
            baseline_reference_template,
            baseline_structure_template,
            rigid_transform,
        )
        positions = torch.as_tensor(
            np.asarray(structure.positions), dtype=torch.float64, device=device
        ).clone().requires_grad_(True)
        assignment = SimpleNamespace(reference_to_source=np.asarray([1, 0]))
        return SimpleNamespace(
            current_positions_reference_order=positions,
            assignment=assignment,
            atom_count=2,
            receipt=_valid_context_receipt(2),
        )

    def energy_components(context):
        x = context.current_positions_reference_order
        quadratic = torch.sum(x * x)
        cubic = torch.sum(x**3)
        index = torch.arange(1, 66, dtype=torch.float64, device=x.device)
        fixed = 0.7 + 0.4 * quadratic + 0.05 * cubic
        parameters = 0.002 * index * quadratic + 0.00001 * index.square() * cubic
        return fixed, parameters

    monkeypatch.setattr(r2r, "_verified_production_context", verified_context)
    monkeypatch.setattr(r2r, "_production_energy_components", energy_components)


def _synthetic_expected(
    positions: np.ndarray, physical_p: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(positions, dtype=torch.float64).clone().requires_grad_(True)
    index = torch.arange(1, 66, dtype=torch.float64)
    p = torch.as_tensor(physical_p, dtype=torch.float64)
    quadratic = torch.sum(x * x)
    cubic = torch.sum(x**3)
    energy = (
        0.7
        + 0.4 * quadratic
        + 0.05 * cubic
        + torch.dot(0.002 * index * quadratic + 0.00001 * index.square() * cubic, p)
    )
    force_reference = -torch.autograd.grad(
        energy, x, create_graph=True, retain_graph=True
    )[0]
    rows = []
    for component in force_reference.reshape(-1):
        rows.append(
            -torch.autograd.grad(component, x, retain_graph=True)[0].reshape(-1)
        )
    hessian_reference = torch.stack(rows)
    component_order = torch.as_tensor([3, 4, 5, 0, 1, 2])
    hessian_source = hessian_reference[
        component_order[:, None], component_order[None, :]
    ]
    force_source = torch.empty_like(force_reference)
    force_source[torch.as_tensor([1, 0])] = force_reference
    return energy, force_source, hessian_source


def test_checkpoint_roundtrip_and_exact_provenance(tmp_path: Path) -> None:
    physical_p = np.linspace(-0.4, 0.7, 65, dtype=np.dtype("<f8"))
    root, receipt_sha = _write_checkpoint(tmp_path / "release", physical_p)
    checkpoint = _load(root, receipt_sha)

    assert checkpoint.physical_p.dtype.str == "<f8"
    assert checkpoint.physical_p.shape == (65,)
    assert checkpoint.physical_p.flags.c_contiguous
    assert not checkpoint.physical_p.flags.writeable
    assert np.array_equal(checkpoint.validated_physical_p(), physical_p)
    assert checkpoint.coefficient_hashes == frozen.physical_p_hashes(physical_p)
    assert checkpoint.receipt["primitive_provenance"] == {
        "module_sha256": frozen.FROZEN_R2R_PRIMITIVE_SHA256,
        "canonical_contract_sha256": frozen.FROZEN_R2R_CANONICAL_SHA256,
        "endpoint_state_sha256": frozen.FROZEN_ENDPOINT_STATE_SHA256,
        "affine_component_names_sha256": (
            frozen.FROZEN_AFFINE_COMPONENT_NAMES_SHA256
        ),
    }
    assert checkpoint.receipt["r2r0_provenance"] == {
        "formal_status": frozen.R2R0_FORMAL_STATUS,
        "aggregate_arrays_sha256": frozen.R2R0_AGGREGATE_ARRAYS_SHA256,
        "aggregate_receipt_sha256": frozen.R2R0_AGGREGATE_RECEIPT_SHA256,
        "launch_receipt_sha256": frozen.R2R0_LAUNCH_RECEIPT_SHA256,
    }
    assert all(value is False for value in checkpoint.receipt["safety"].values())


@pytest.mark.parametrize(
    "bad_value,match",
    [
        (np.zeros(65, dtype=np.float32), "little-endian FP64"),
        (np.zeros(64, dtype=np.dtype("<f8")), "exactly 65"),
        (
            np.r_[np.zeros(64), np.nan].astype(np.dtype("<f8")),
            "non-finite",
        ),
    ],
)
def test_checkpoint_rejects_bad_physical_p_npz(
    tmp_path: Path, bad_value: np.ndarray, match: str
) -> None:
    root, receipt_sha = _write_checkpoint(
        tmp_path / "release", np.zeros(65, dtype=np.dtype("<f8"))
    )
    np.savez(root / frozen.CHECKPOINT_ARRAY_BASENAME, physical_p=bad_value)
    with pytest.raises(ValueError, match=match):
        _load(root, receipt_sha)


def test_checkpoint_rejects_extra_npz_member_and_release_file(tmp_path: Path) -> None:
    root, receipt_sha = _write_checkpoint(
        tmp_path / "release", np.zeros(65, dtype=np.dtype("<f8"))
    )
    np.savez(
        root / frozen.CHECKPOINT_ARRAY_BASENAME,
        physical_p=np.zeros(65, dtype=np.dtype("<f8")),
        intercept=np.zeros(1, dtype=np.dtype("<f8")),
    )
    with pytest.raises(ValueError, match="only physical_p"):
        _load(root, receipt_sha)

    root2, receipt_sha2 = _write_checkpoint(
        tmp_path / "release_extra", np.zeros(65, dtype=np.dtype("<f8"))
    )
    (root2 / "extra.txt").write_text("unexpected\n", encoding="ascii")
    with pytest.raises(ValueError, match="file set changed"):
        _load(root2, receipt_sha2)


@pytest.mark.parametrize(
    "restricted_name",
    [
        "holdout_release",
        "Hold OutRelease",
        "seed_1_release",
        "seed-2-release",
        "s.e.e.d.1.release",
        "s.e.e.d.0.2.release",
        "seed01Release",
        "seed002Release",
        "small_release",
        "S.M.A.L.L.Release",
        "h-e-l-d release",
        "s_u_p_p_o_r_tRelease",
        "ReSeRvEdRelease",
    ],
)
def test_checkpoint_rejects_restricted_path_tokens_before_open(
    tmp_path: Path,
    restricted_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_count = 0

    def unexpected_open(*args, **kwargs):
        nonlocal open_count
        del args, kwargs
        open_count += 1
        raise AssertionError("restricted checkpoint path reached os.open")

    monkeypatch.setattr(frozen.os, "open", unexpected_open)
    with pytest.raises(ValueError, match="forbidden checkpoint path token"):
        _load(tmp_path / restricted_name, "0" * 64)
    assert open_count == 0


def test_parent_symlink_swap_after_validation_never_opens_external_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_parent = tmp_path / "trusted_parent"
    trusted_parent.mkdir()
    root, receipt_sha = _write_checkpoint(
        trusted_parent / "release", np.zeros(65, dtype=np.dtype("<f8"))
    )
    external_parent = tmp_path / "external_parent"
    external_parent.mkdir()
    external_root, _ = _write_checkpoint(
        external_parent / "release", np.ones(65, dtype=np.dtype("<f8"))
    )
    displaced_parent = tmp_path / "displaced_trusted_parent"

    def identity(path: Path) -> tuple[int, int]:
        observed = path.stat()
        return observed.st_dev, observed.st_ino

    external_identities = {identity(external_parent), identity(external_root)}
    external_identities.update(
        identity(external_root / name) for name in frozen.CHECKPOINT_FILE_SET
    )
    real_safe_root = frozen._safe_checkpoint_root
    real_open = os.open
    external_open_count = 0
    swapped = False

    def validate_then_swap(checkpoint_root: Path) -> Path:
        nonlocal swapped
        lexical_root = real_safe_root(checkpoint_root)
        trusted_parent.rename(displaced_parent)
        trusted_parent.symlink_to(external_parent, target_is_directory=True)
        swapped = True
        return lexical_root

    def counted_open(
        path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal external_open_count
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) in external_identities:
            external_open_count += 1
        return descriptor

    monkeypatch.setattr(frozen, "_safe_checkpoint_root", validate_then_swap)
    monkeypatch.setattr(frozen.os, "open", counted_open)
    with pytest.raises(ValueError, match="path component"):
        _load(root, receipt_sha)
    assert swapped is True
    assert external_open_count == 0


def test_parent_directory_swap_after_validation_never_opens_external_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_parent = tmp_path / "trusted_directory_parent"
    trusted_parent.mkdir()
    root, receipt_sha = _write_checkpoint(
        trusted_parent / "release", np.zeros(65, dtype=np.dtype("<f8"))
    )
    external_parent = tmp_path / "external_directory_parent"
    external_parent.mkdir()
    external_root, _ = _write_checkpoint(
        external_parent / "release", np.ones(65, dtype=np.dtype("<f8"))
    )
    displaced_parent = tmp_path / "displaced_directory_parent"

    def identity(path: Path) -> tuple[int, int]:
        observed = path.stat()
        return observed.st_dev, observed.st_ino

    external_member_identities = {
        identity(external_root / name) for name in frozen.CHECKPOINT_FILE_SET
    }
    real_safe_root = frozen._safe_checkpoint_root
    real_open = os.open
    real_read = os.read
    external_member_open_count = 0
    external_member_read_count = 0
    swapped = False

    def validate_then_swap(checkpoint_root: Path) -> frozen._BoundCheckpointRoot:
        nonlocal swapped
        bound_root = real_safe_root(checkpoint_root)
        trusted_parent.rename(displaced_parent)
        external_parent.rename(trusted_parent)
        swapped = True
        return bound_root

    def counted_open(
        path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal external_member_open_count
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) in external_member_identities:
            external_member_open_count += 1
        return descriptor

    def counted_read(descriptor: int, size: int) -> bytes:
        nonlocal external_member_read_count
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) in external_member_identities:
            external_member_read_count += 1
        return real_read(descriptor, size)

    monkeypatch.setattr(frozen, "_safe_checkpoint_root", validate_then_swap)
    monkeypatch.setattr(frozen.os, "open", counted_open)
    monkeypatch.setattr(frozen.os, "read", counted_read)
    with pytest.raises(ValueError, match="component identity"):
        _load(root, receipt_sha)
    assert swapped is True
    assert external_member_open_count == 0
    assert external_member_read_count == 0


def test_checkpoint_rejects_hardlink_and_broken_symlink_members(
    tmp_path: Path,
) -> None:
    root, receipt_sha = _write_checkpoint(
        tmp_path / "release_hardlink", np.zeros(65, dtype=np.dtype("<f8"))
    )
    os.link(
        root / frozen.CHECKPOINT_ARRAY_BASENAME,
        tmp_path / "checkpoint_array_alias.npz",
    )
    with pytest.raises(ValueError, match="one hard link"):
        _load(root, receipt_sha)

    root2, receipt_sha2 = _write_checkpoint(
        tmp_path / "release_symlink", np.zeros(65, dtype=np.dtype("<f8"))
    )
    arrays_path = root2 / frozen.CHECKPOINT_ARRAY_BASENAME
    arrays_path.unlink()
    arrays_path.symlink_to(tmp_path / "missing_checkpoint_arrays.npz")
    with pytest.raises(ValueError, match="ordinary file"):
        _load(root2, receipt_sha2)


def test_checkpoint_rejects_receipt_marker_and_external_hash_tamper(
    tmp_path: Path,
) -> None:
    root, receipt_sha = _write_checkpoint(
        tmp_path / "release", np.zeros(65, dtype=np.dtype("<f8"))
    )
    with pytest.raises(ValueError, match="externally frozen"):
        _load(root, "f" * 64)

    (root / frozen.CHECKPOINT_MARKER_BASENAME).write_text(
        "wrong\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="marker"):
        _load(root, receipt_sha)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda receipt: receipt["primitive_provenance"].__setitem__(
                "module_sha256", "0" * 64
            ),
            "primitive/endpoint provenance",
        ),
        (
            lambda receipt: receipt["r2r0_provenance"].__setitem__(
                "aggregate_arrays_sha256", "0" * 64
            ),
            "attempt3 provenance",
        ),
        (
            lambda receipt: receipt["safety"].__setitem__(
                "energy_labels_used", True
            ),
            "safety fields",
        ),
        (
            lambda receipt: receipt["coefficient_layout"].__setitem__(
                "global_intercept_present", True
            ),
            "coefficient layout",
        ),
        (
            lambda receipt: receipt.__setitem__("unrelated", False),
            "key set changed",
        ),
    ],
)
def test_self_consistent_receipt_rewrite_still_fails_closed(
    tmp_path: Path, mutation, match: str
) -> None:
    root, _ = _write_checkpoint(
        tmp_path / "release", np.zeros(65, dtype=np.dtype("<f8"))
    )
    rewritten_sha = _rewrite_receipt(root, mutation)
    with pytest.raises(ValueError, match=match):
        _load(root, rewritten_sha)


def test_synthetic_scalar_first_efh_and_receipt_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_synthetic_context(monkeypatch)
    physical_p = np.linspace(-0.2, 0.3, 65, dtype=np.dtype("<f8"))
    root, receipt_sha = _write_checkpoint(tmp_path / "release", physical_p)
    checkpoint = _load(root, receipt_sha)
    structure = Atoms(
        "CC",
        positions=[[0.2, -0.1, 0.3], [0.5, 0.4, -0.2]],
        cell=np.eye(3) * 8.0,
        pbc=False,
    )
    reference = structure.copy()
    expected_energy, expected_force, expected_hessian = _synthetic_expected(
        np.asarray(structure.positions), physical_p
    )

    probe = frozen.production_frozen_readout_energy_force(
        torch.nn.Identity(), structure, reference, checkpoint, device="cpu"
    )
    mechanics = frozen.production_frozen_readout_energy_force_hessian(
        torch.nn.Identity(), structure, reference, checkpoint, device="cpu"
    )
    assert torch.allclose(probe.energy_eV, expected_energy, atol=1.0e-14, rtol=0.0)
    assert torch.allclose(
        probe.force_source_order_eV_A, expected_force, atol=1.0e-14, rtol=0.0
    )
    assert torch.allclose(mechanics.energy_eV, expected_energy, atol=1.0e-14, rtol=0.0)
    assert torch.allclose(
        mechanics.force_source_order_eV_A,
        expected_force,
        atol=1.0e-14,
        rtol=0.0,
    )
    assert torch.allclose(
        mechanics.Hessian_source_order_eV_A2,
        expected_hessian,
        atol=1.0e-13,
        rtol=0.0,
    )
    assert mechanics.Hessian_antisymmetry_max_abs_eV_A2 < 1.0e-14
    receipt = probe.query_receipt
    assert receipt["single_scalar_before_autograd"] is True
    assert receipt["fixed_carrier_coefficient"] == 1.0
    assert receipt["global_intercept_present"] is False
    assert receipt["old_R2Q_tail_added"] is False
    assert receipt["checkpoint_receipt_sha256"] == receipt_sha
    assert receipt["endpoint_state_sha256"] == frozen.FROZEN_ENDPOINT_STATE_SHA256
    assert receipt["reference_semantic_sha256"] == "3" * 64
    assert receipt["formal_R2O_graph_semantic_sha256"] == "4" * 64
    assert receipt["assignment_and_MIC_semantic_sha256"] == "5" * 64
    assert receipt["physical_p_semantic_sha256"] == frozen.physical_p_hashes(
        physical_p
    )["physical_p_semantic_sha256"]


def test_every_call_rehashes_memory_file_and_live_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_synthetic_context(monkeypatch)
    root, receipt_sha = _write_checkpoint(
        tmp_path / "release", np.zeros(65, dtype=np.dtype("<f8"))
    )
    checkpoint = _load(root, receipt_sha)
    structure = Atoms("CC", positions=np.zeros((2, 3)), cell=np.eye(3) * 8.0)

    checkpoint.physical_p.setflags(write=True)
    with pytest.raises(ValueError, match="write protection"):
        frozen.production_frozen_readout_energy_force(
            torch.nn.Identity(), structure, structure, checkpoint, device="cpu"
        )
    checkpoint.physical_p[0] = 1.0
    checkpoint.physical_p.setflags(write=False)
    with pytest.raises(ValueError, match="in-memory"):
        frozen.production_frozen_readout_energy_force(
            torch.nn.Identity(), structure, structure, checkpoint, device="cpu"
        )

    root2, receipt_sha2 = _write_checkpoint(
        tmp_path / "release_file", np.zeros(65, dtype=np.dtype("<f8"))
    )
    checkpoint2 = _load(root2, receipt_sha2)
    np.savez(
        root2 / frozen.CHECKPOINT_ARRAY_BASENAME,
        physical_p=np.ones(65, dtype=np.dtype("<f8")),
    )
    with pytest.raises(ValueError, match="coefficient NPZ|bind"):
        frozen.production_frozen_readout_energy_force(
            torch.nn.Identity(), structure, structure, checkpoint2, device="cpu"
        )

    root3, receipt_sha3 = _write_checkpoint(
        tmp_path / "release_primitive", np.zeros(65, dtype=np.dtype("<f8"))
    )
    checkpoint3 = _load(root3, receipt_sha3)
    monkeypatch.setattr(r2r, "LINEAR_DESIGN_WIDTH", 64)
    with pytest.raises(ValueError, match="parameter width"):
        frozen.production_frozen_readout_energy_force(
            torch.nn.Identity(), structure, structure, checkpoint3, device="cpu"
        )


def test_live_primitive_rejects_hardlinked_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primitive_copy = tmp_path / "primitive_copy.py"
    primitive_copy.write_bytes(Path(r2r.__file__).read_bytes())
    os.link(primitive_copy, tmp_path / "primitive_alias.py")
    monkeypatch.setattr(r2r, "__file__", str(primitive_copy))
    with pytest.raises(ValueError, match="one hard link"):
        frozen._validate_live_primitive()


def test_live_primitive_parent_symlink_swap_never_opens_external_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_parent = tmp_path / "trusted_primitive_parent"
    trusted_parent.mkdir()
    trusted_source = trusted_parent / "primitive.py"
    trusted_source.write_bytes(Path(r2r.__file__).read_bytes())
    external_parent = tmp_path / "external_primitive_parent"
    external_parent.mkdir()
    external_source = external_parent / "primitive.py"
    external_source.write_bytes(b"external primitive must never be opened\n")
    displaced_parent = tmp_path / "displaced_primitive_parent"

    def identity(path: Path) -> tuple[int, int]:
        observed = path.stat()
        return observed.st_dev, observed.st_ino

    external_identities = {identity(external_parent), identity(external_source)}
    real_lexical_path = frozen._lexical_absolute_file_path
    real_open = os.open
    external_open_count = 0
    swapped = False

    def normalize_then_swap(path: Path, label: str) -> Path:
        nonlocal swapped
        lexical_path = real_lexical_path(path, label)
        trusted_parent.rename(displaced_parent)
        trusted_parent.symlink_to(external_parent, target_is_directory=True)
        swapped = True
        return lexical_path

    def counted_open(
        path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal external_open_count
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) in external_identities:
            external_open_count += 1
        return descriptor

    monkeypatch.setattr(r2r, "__file__", str(trusted_source))
    monkeypatch.setattr(
        frozen, "_lexical_absolute_file_path", normalize_then_swap
    )
    monkeypatch.setattr(frozen.os, "open", counted_open)
    with pytest.raises(ValueError, match="path component"):
        frozen._validate_live_primitive()
    assert swapped is True
    assert external_open_count == 0


def test_live_primitive_final_rebind_stops_before_external_source_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_parent = tmp_path / "trusted_primitive_directory"
    trusted_parent.mkdir()
    trusted_source = trusted_parent / "primitive.py"
    trusted_source.write_bytes(Path(r2r.__file__).read_bytes())
    external_parent = tmp_path / "external_primitive_directory"
    external_parent.mkdir()
    external_source = external_parent / "primitive.py"
    external_source.write_bytes(b"external source must not be opened or read\n")
    displaced_parent = tmp_path / "displaced_primitive_directory"
    external_stat = external_source.stat()
    external_identity = (external_stat.st_dev, external_stat.st_ino)

    real_file_open = frozen._open_lexical_regular_file_fd
    real_open = os.open
    real_read = os.read
    walker_call_count = 0
    external_open_count = 0
    external_read_count = 0
    swapped = False

    def swap_before_final_rebind(path: Path, label: str, **kwargs):
        nonlocal walker_call_count, swapped
        walker_call_count += 1
        if walker_call_count == 2:
            trusted_parent.rename(displaced_parent)
            external_parent.rename(trusted_parent)
            swapped = True
        return real_file_open(path, label, **kwargs)

    def counted_open(
        path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal external_open_count
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == external_identity:
            external_open_count += 1
        return descriptor

    def counted_read(descriptor: int, size: int) -> bytes:
        nonlocal external_read_count
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == external_identity:
            external_read_count += 1
        return real_read(descriptor, size)

    monkeypatch.setattr(r2r, "__file__", str(trusted_source))
    monkeypatch.setattr(
        frozen, "_open_lexical_regular_file_fd", swap_before_final_rebind
    )
    monkeypatch.setattr(frozen.os, "open", counted_open)
    monkeypatch.setattr(frozen.os, "read", counted_read)
    with pytest.raises(ValueError, match="component identity"):
        frozen._validate_live_primitive()
    assert walker_call_count == 2
    assert swapped is True
    assert external_open_count == 0
    assert external_read_count == 0


def test_checkpoint_hash_and_parse_use_the_same_snapshot_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, receipt_sha = _write_checkpoint(
        tmp_path / "release", np.zeros(65, dtype=np.dtype("<f8"))
    )
    original_snapshot = frozen._read_checkpoint_release_bytes
    snapshot_count = 0

    def snapshot_then_replace(
        release_root: frozen._BoundCheckpointRoot,
    ) -> dict[str, bytes]:
        nonlocal snapshot_count
        snapshot = original_snapshot(release_root)
        if snapshot_count == 0:
            np.savez(
                release_root.path / frozen.CHECKPOINT_ARRAY_BASENAME,
                physical_p=np.ones(65, dtype=np.dtype("<f8")),
            )
        snapshot_count += 1
        return snapshot

    monkeypatch.setattr(
        frozen, "_read_checkpoint_release_bytes", snapshot_then_replace
    )
    checkpoint = _load(root, receipt_sha)
    assert np.array_equal(
        checkpoint.physical_p, np.zeros(65, dtype=np.dtype("<f8"))
    )
    with pytest.raises(ValueError, match="bind"):
        checkpoint.validated_physical_p()


def test_real_endpoint_canonical_and_arbitrary_ef_and_canonical_hessian_parity(
    tmp_path: Path,
) -> None:
    # The formal geometry-only parser converts no force/energy label columns.
    thermal = formal.read_geometry_only_extxyz(DATA / "train_thermal.xyz")[0]
    reference = read(DATA / "reference_6x6.xyz", index=0)
    endpoint = r2o.torch_load(RUN / "endpoint.pt", map_location="cpu")
    model = endpoint["model"].to(dtype=torch.float64).eval()
    assert r2o.state_dict_sha256(model) == frozen.FROZEN_ENDPOINT_STATE_SHA256
    canonical_constant_before = tuple(r2r.MECHANICS_PROBE_COEFFICIENTS)
    canonical_hash_before = r2r.MECHANICS_PROBE_COEFFICIENTS_SHA256

    canonical_p = np.asarray(
        r2r.MECHANICS_PROBE_COEFFICIENTS, dtype=np.dtype("<f8")
    )
    canonical_root, canonical_receipt_sha = _write_checkpoint(
        tmp_path / "canonical_release", canonical_p
    )
    canonical_checkpoint = _load(canonical_root, canonical_receipt_sha)
    new_canonical = frozen.production_frozen_readout_energy_force(
        model, thermal, reference, canonical_checkpoint, device="cpu"
    )
    old_canonical = r2r.production_combined_energy_force(
        model, thermal, reference, device="cpu"
    )
    design = r2r.production_linear_design_query(
        model, thermal, reference, device="cpu"
    )
    design_canonical = design.canonical_combined_probe()
    assert abs(float((new_canonical.energy_eV - old_canonical.energy_eV).detach())) < 1.0e-12
    assert float(
        torch.max(
            torch.abs(
                new_canonical.force_source_order_eV_A
                - old_canonical.force_source_order_eV_A
            )
        ).detach()
    ) < 1.0e-11
    assert abs(
        float((new_canonical.energy_eV - design_canonical.energy_eV).detach())
    ) < 1.0e-12
    assert float(
        torch.max(
            torch.abs(
                new_canonical.force_source_order_eV_A
                - design_canonical.force_source_order_eV_A
            )
        ).detach()
    ) < 1.0e-11

    arbitrary_p = np.linspace(-0.11, 0.17, 65, dtype=np.dtype("<f8"))
    arbitrary_root, arbitrary_receipt_sha = _write_checkpoint(
        tmp_path / "arbitrary_release", arbitrary_p
    )
    arbitrary_checkpoint = _load(arbitrary_root, arbitrary_receipt_sha)
    arbitrary = frozen.production_frozen_readout_energy_force(
        model, thermal, reference, arbitrary_checkpoint, device="cpu"
    )
    arbitrary_tensor = torch.as_tensor(arbitrary_p, dtype=torch.float64)
    expected_energy = design.fixed_offset_energy_eV[0] + torch.dot(
        design.parameter_energy_design_eV, arbitrary_tensor
    )
    expected_force = (
        design.fixed_offset_force_source_order_eV_A[:, 0]
        + design.parameter_force_design_source_order_eV_A @ arbitrary_tensor
    ).reshape(72, 3)
    assert abs(float((arbitrary.energy_eV - expected_energy).detach())) < 1.0e-12
    assert float(
        torch.max(torch.abs(arbitrary.force_source_order_eV_A - expected_force)).detach()
    ) < 1.0e-11
    assert arbitrary.query_receipt["endpoint_state_sha256"] == (
        frozen.FROZEN_ENDPOINT_STATE_SHA256
    )
    assert arbitrary.query_receipt["formal_R2O_graph_semantic_sha256"] == (
        design.receipt["formal_R2O_graph_semantic_sha256"]
    )
    assert arbitrary.query_receipt["reference_semantic_sha256"] == (
        design.receipt["reference_semantic_sha256"]
    )
    assert arbitrary.query_receipt["assignment_and_MIC_semantic_sha256"] == (
        design.receipt["assignment_and_MIC_semantic_sha256"]
    )

    # Full-H parity is evaluated at the actual frozen reference, where both
    # APIs still construct and differentiate the complete scalar graph.
    new_hessian = frozen.production_frozen_readout_energy_force_hessian(
        model, reference.copy(), reference, canonical_checkpoint, device="cpu"
    )
    old_hessian = r2r.production_combined_energy_force_hessian(
        model, reference.copy(), reference, device="cpu"
    )
    assert abs(float((new_hessian.energy_eV - old_hessian.energy_eV).detach())) < 1.0e-12
    assert float(
        torch.max(
            torch.abs(
                new_hessian.force_source_order_eV_A
                - old_hessian.force_source_order_eV_A
            )
        ).detach()
    ) < 1.0e-11
    assert float(
        torch.max(
            torch.abs(
                new_hessian.Hessian_source_order_eV_A2
                - old_hessian.Hessian_source_order_eV_A2
            )
        ).detach()
    ) < 1.0e-10

    assert tuple(r2r.MECHANICS_PROBE_COEFFICIENTS) == canonical_constant_before
    assert r2r.MECHANICS_PROBE_COEFFICIENTS_SHA256 == canonical_hash_before

    tampered = r2o.torch_load(RUN / "endpoint.pt", map_location="cpu")["model"].to(
        dtype=torch.float64
    ).eval()
    with torch.no_grad():
        next(tampered.parameters()).reshape(-1)[0].add_(1.0e-8)
    with pytest.raises(ValueError, match="frozen endpoint state"):
        frozen.production_frozen_readout_energy_force(
            tampered, thermal, reference, arbitrary_checkpoint, device="cpu"
        )
