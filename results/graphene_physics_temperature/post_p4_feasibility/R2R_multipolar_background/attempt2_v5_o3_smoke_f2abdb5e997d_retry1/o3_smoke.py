#!/usr/bin/env python3
"""Bounded, diagnostic-only R2R-0 O(3) smoke on one train geometry."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read


FORMAT = "graphene_r2r0_attempt2_v5_o3_smoke_v1"
EXPECTED_MANIFEST_SHA256 = (
    "f2abdb5e997db69e0f406458c2dd3b0331e22d0e5d8828b4986b6f964c8753a6"
)
EXPECTED_SOURCES = {
    "scripts/smearing_kink/graphene_r2r_multipolar_background.py": (
        "4db3665f945783086de5c7a2f6262843bbe7d03df2a32f9153e450200b25064c"
    ),
    "scripts/smearing_kink/graphene_r2r0_formal.py": (
        "813ccffaf06e2027ce42e4aa7685fd2f1552da518a9bd3574edead5af8bafbfb"
    ),
    "scripts/smearing_kink/graphene_r2o_taylor_null.py": (
        "0eedea5a59b8f717559feda97b2c2956dd6f274fe4963de10c214db53b4610e2"
    ),
    "scripts/smearing_kink/evaluate_graphene_r2o_taylor_null.py": (
        "14cf2a160a9b0281c0c2149caec6d62f9a31e40a9c7af39ea3ea2e14956c79ce"
    ),
    "scripts/smearing_kink/run_graphene_r2r0_formal.py": (
        "f28961e25c235b47d0ac8d6bca3cdb52fd52ed832e5cabeeda72d08dcf7ba316"
    ),
    "scripts/smearing_kink/aggregate_graphene_r2r0_formal.py": (
        "29b35a7899cd02682d2575dbbeb9bfefb19eaec289f5a09fd82c1b06b4827abf"
    ),
    "scripts/smearing_kink/launch_graphene_r2r0_formal.py": (
        "7c9807c088e4f66b9d7286d8ec79e0f517dff405065a9eb7bc7aff6b57d09467"
    ),
    "tests/test_graphene_r2r_multipolar_background.py": (
        "1255c5720cb0fe575fe7edd9d211035799feccebd922488bc33ebe20f7c36c08"
    ),
    "tests/test_graphene_r2r0_formal.py": (
        "be95450fb05ad5337833e153e67c7b9a2e9c6b3b1f959cc74c4ae646dbf7c780"
    ),
    "docs/GRAPHENE_R2R_MULTIPOLAR_BACKGROUND_CONTRACT_2026-08-25.md": (
        "003f4c3e949e6432b56a350541ca15e7bde48749ad2150129e8ea227cf5c56c9"
    ),
    "docs/GRAPHENE_R2R0_FORMAL_CONTRACT_2026-08-25.md": (
        "f5818c84c5135d6c8cf91ce4dc8228c5e6d5b952ee855a43e18634a0999dfc45"
    ),
}
EXPECTED_INPUTS = {
    "endpoint.pt": "31dd053a01c700ee73eedff35e85c02cf7765de2299ff418259c270b3f823eb5",
    "endpoint_receipt.json": (
        "810a45ab66a885a940e086a7be9c4304050a0772d477ecd32bd03c20c8ef43a1"
    ),
    "ENDPOINT_FROZEN": (
        "5fa7771f4c8f7d32990e4bb0d0350e8f1f020c1685030626ac8555b06a4ce0ed"
    ),
    "reference_6x6.xyz": (
        "2793b514768c6d875831a1b62fcb2aa5f858830cb2de7b1b2d5492dabeb8da90"
    ),
    "train_thermal.xyz": (
        "cc2e9c68d418fba996e8ac89c8156c8adf00768406f6171aaa57f0a9bd2720c1"
    ),
}
FORBIDDEN_PATH_TOKENS = (
    "seed1",
    "seed2",
    "held",
    "support",
    "outer_fold",
    "reserved",
    "valid_e50",
    "small_gate",
    "full_report",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def assert_no_symlink(path: Path, stop: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"symlink forbidden in smoke path: {current}")
        if current == stop:
            break
        if stop not in current.parents:
            raise ValueError(f"path escapes bundle: {path}")
        current = current.parent


def orthogonal_pair() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(83)
    proper, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(proper) < 0.0:
        proper[:, 0] *= -1.0
    improper = proper.copy()
    improper[:, 0] *= -1.0
    return proper, improper


def transformed_atoms(source, matrix: np.ndarray):
    result = source.copy()
    result.positions = np.asarray(source.positions, dtype=np.float64) @ matrix.T
    result.set_cell(
        np.asarray(source.cell, dtype=np.float64) @ matrix.T,
        scale_atoms=False,
    )
    return result


def graph_covariance_receipt(
    base: dict,
    covariant: dict,
    native_rebuilt: dict,
    matrix: np.ndarray,
) -> dict:
    vector_differences = {}
    for key in ("positions", "shifts"):
        expected = base[key].detach().cpu().numpy().astype(np.float64) @ matrix.T
        observed = covariant[key].detach().cpu().numpy().astype(np.float64)
        vector_differences[key] = float(np.max(np.abs(observed - expected)))
    base_cell = base["cell"].detach().cpu().numpy().astype(np.float64)
    transformed_cell = covariant["cell"].detach().cpu().numpy().astype(np.float64)
    vector_differences["cell"] = float(
        np.max(np.abs(transformed_cell - base_cell @ matrix.T))
    )
    discrete_keys = (
        "edge_index",
        "unit_shifts",
        "node_attrs",
        "batch",
        "ptr",
        "head",
        "pbc",
    )
    exact = {
        key: bool(torch.equal(base[key].detach().cpu(), covariant[key].detach().cpu()))
        for key in discrete_keys
    }
    native_minus_covariant = {
        key: float(
            torch.max(torch.abs(native_rebuilt[key] - covariant[key])).detach().cpu()
        )
        for key in ("positions", "shifts", "cell")
    }
    return {
        "covariant_vector_max_abs_difference_A": vector_differences,
        "covariant_unchanged_tensor_exact": exact,
        "covariant_all_invariant_tensors_exact": all(exact.values()),
        "native_rebuild_minus_covariant_vector_tensor_max_abs": (
            native_minus_covariant
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = args.bundle.resolve(strict=True)
    output = args.output
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"smoke output must be fresh: {output}")
    if output.parent.resolve(strict=True) != bundle.parent.resolve(strict=True):
        raise ValueError("smoke output must be a sibling of the immutable bundle")
    output.mkdir(mode=0o700)
    running = output / "RUNNING"
    atomic_text(running, FORMAT + "\n")

    try:
        for path in [bundle, *bundle.rglob("*")]:
            assert_no_symlink(path, bundle)

        sources = bundle / "sources"
        scripts = sources / "scripts" / "smearing_kink"
        inputs = bundle / "inputs"
        manifest_path = bundle / "control" / "freeze_manifest.json"
        driver_path = Path(__file__).resolve(strict=True)
        if driver_path != bundle / "o3_smoke.py":
            raise ValueError("smoke driver import path differs from staged snapshot")

        source_hashes = {
            relative: sha256(sources / relative) for relative in EXPECTED_SOURCES
        }
        input_hashes = {name: sha256(inputs / name) for name in EXPECTED_INPUTS}
        if source_hashes != EXPECTED_SOURCES:
            raise ValueError(f"staged smoke source hashes changed: {source_hashes}")
        if input_hashes != EXPECTED_INPUTS:
            raise ValueError(f"staged smoke input hashes changed: {input_hashes}")
        if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
            raise ValueError("staged attempt2 manifest hash changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("authorization_marker_created") is not False:
            raise ValueError("diagnostic smoke requires a manifest without formal GO")
        if manifest.get("can_authorize_fit_or_training") is not False:
            raise ValueError("diagnostic manifest unexpectedly authorizes fitting")
        manifest_sources = manifest["formal_source_sha256"] | manifest[
            "frozen_primitive_sha256"
        ]
        required_manifest_sources = {
            "formal_core": EXPECTED_SOURCES[
                "scripts/smearing_kink/graphene_r2r0_formal.py"
            ],
            "shard_cli": EXPECTED_SOURCES[
                "scripts/smearing_kink/run_graphene_r2r0_formal.py"
            ],
            "aggregate_cli": EXPECTED_SOURCES[
                "scripts/smearing_kink/aggregate_graphene_r2r0_formal.py"
            ],
            "launcher": EXPECTED_SOURCES[
                "scripts/smearing_kink/launch_graphene_r2r0_formal.py"
            ],
            "tests": EXPECTED_SOURCES["tests/test_graphene_r2r0_formal.py"],
            "contract_doc": EXPECTED_SOURCES[
                "docs/GRAPHENE_R2R0_FORMAL_CONTRACT_2026-08-25.md"
            ],
            "R2R_module": EXPECTED_SOURCES[
                "scripts/smearing_kink/graphene_r2r_multipolar_background.py"
            ],
            "R2R_tests": EXPECTED_SOURCES[
                "tests/test_graphene_r2r_multipolar_background.py"
            ],
            "R2R_contract_doc": EXPECTED_SOURCES[
                "docs/GRAPHENE_R2R_MULTIPOLAR_BACKGROUND_CONTRACT_2026-08-25.md"
            ],
            "R2O_wrapper": EXPECTED_SOURCES[
                "scripts/smearing_kink/graphene_r2o_taylor_null.py"
            ],
            "R2O_evaluator": EXPECTED_SOURCES[
                "scripts/smearing_kink/evaluate_graphene_r2o_taylor_null.py"
            ],
        }
        for key, expected in required_manifest_sources.items():
            if manifest_sources.get(key) != expected:
                raise ValueError(f"manifest source hash changed: {key}")
        manifest_input_keys = {
            "endpoint.pt": "endpoint_checkpoint",
            "endpoint_receipt.json": "endpoint_receipt",
            "ENDPOINT_FROZEN": "endpoint_marker",
            "reference_6x6.xyz": "reference_6x6",
            "train_thermal.xyz": "thermal92",
        }
        for basename, role in manifest_input_keys.items():
            if manifest["input_sha256"].get(role) != EXPECTED_INPUTS[basename]:
                raise ValueError(f"manifest input hash changed: {role}")

        lexical_paths = [str(path).lower() for path in inputs.iterdir()]
        if any(token in value for token in FORBIDDEN_PATH_TOKENS for value in lexical_paths):
            raise ValueError("held/support/forbidden path token reached diagnostic smoke")

        sys.path.insert(0, str(scripts))
        import evaluate_graphene_r2o_taylor_null as r2o_evaluator
        import graphene_r2o_taylor_null as r2o
        import graphene_r2r0_formal as formal
        import graphene_r2r_multipolar_background as r2r

        imported = {
            "scripts/smearing_kink/graphene_r2r_multipolar_background.py": Path(
                r2r.__file__
            ).resolve(),
            "scripts/smearing_kink/graphene_r2r0_formal.py": Path(
                formal.__file__
            ).resolve(),
            "scripts/smearing_kink/graphene_r2o_taylor_null.py": Path(
                r2o.__file__
            ).resolve(),
            "scripts/smearing_kink/evaluate_graphene_r2o_taylor_null.py": Path(
                r2o_evaluator.__file__
            ).resolve(),
        }
        for relative, path in imported.items():
            if path != (sources / relative).resolve(strict=True):
                raise ValueError(f"module imported outside staged snapshot: {relative}")
            if sha256(path) != EXPECTED_SOURCES[relative]:
                raise ValueError(f"imported module hash changed: {relative}")
        if formal.FROZEN_R2R_MODULE_SHA256 != EXPECTED_SOURCES[
            "scripts/smearing_kink/graphene_r2r_multipolar_background.py"
        ]:
            raise ValueError("formal core is not bound to the signed-zero R2R snapshot")
        if formal.FORMAL_CONTRACT_SHA256 != manifest["formal_contract_sha256"]:
            raise ValueError("staged formal contract semantic hash differs from manifest")
        if formal.validate_frozen_sources() != manifest["frozen_primitive_sha256"]:
            raise ValueError("formal primitive source closure differs from manifest")
        live_formal_sources = {
            name: formal.sha256(path)
            for name, path in formal.FORMAL_SOURCE_PATHS.items()
        }
        if live_formal_sources != manifest["formal_source_sha256"]:
            raise ValueError("formal source closure differs from manifest")

        endpoint_receipt_path = inputs / "endpoint_receipt.json"
        endpoint_receipt = json.loads(endpoint_receipt_path.read_text(encoding="utf-8"))
        endpoint_receipt_required = {
            "format": "graphene_r2q_frozen_endpoint_v1",
            "status": "FOUR_STEP_ENDPOINT_FROZEN",
            "endpoint_checkpoint_sha256": EXPECTED_INPUTS["endpoint.pt"],
            "endpoint_model_state_sha256": r2r.R2Q_ENDPOINT_STATE_SHA256,
        }
        for key, expected in endpoint_receipt_required.items():
            if endpoint_receipt.get(key) != expected:
                raise ValueError(f"endpoint receipt field changed: {key}")
        if (inputs / "ENDPOINT_FROZEN").read_bytes() != (
            EXPECTED_INPUTS["endpoint_receipt.json"] + "\n"
        ).encode("ascii"):
            raise ValueError("endpoint marker does not bind the exact receipt")

        checkpoint = r2o.torch_load(inputs / "endpoint.pt", map_location="cpu")
        model = checkpoint.get("model")
        if not isinstance(model, torch.nn.Module):
            raise ValueError("endpoint checkpoint lacks complete model object")
        default_dtype_before = str(torch.get_default_dtype())
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("production graph default dtype must begin as torch.float32")
        model = model.to(device="cuda", dtype=torch.float64).eval()
        r2o.validate_mace_architecture(model)
        state_before = r2o.state_dict_sha256(model)
        if state_before != r2r.R2Q_ENDPOINT_STATE_SHA256:
            raise ValueError("endpoint state differs from frozen R2Q state")

        thermal = formal.read_geometry_only_extxyz(inputs / "train_thermal.xyz")
        if len(thermal) != 92:
            raise ValueError("thermal train structure count changed")
        probe = thermal[0]
        if probe.info.get("config_type") != "r2o_exact_e50_seed0_train":
            raise ValueError("thermal0 identity changed")
        if probe.calc is not None or set(probe.arrays) != {"numbers", "positions"}:
            raise ValueError("geometry-only loader attached a label")
        reference = read(inputs / "reference_6x6.xyz", index=0)
        base_reference_hash = r2r.validate_reference_semantics(reference)

        public_callable = r2r.production_combined_energy_force
        if (
            public_callable.__name__ != "production_combined_energy_force"
            or public_callable.__module__ != "graphene_r2r_multipolar_background"
            or Path(inspect.getsourcefile(public_callable)).resolve()
            != imported["scripts/smearing_kink/graphene_r2r_multipolar_background.py"]
        ):
            raise ValueError("combined public production callable was replaced")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        call_count = 0
        states = [state_before]
        default_dtypes = [default_dtype_before]

        def combined(structure, reference_template, **kwargs):
            nonlocal call_count
            call_count += 1
            value = public_callable(
                model,
                structure,
                reference_template,
                device="cuda",
                **kwargs,
            )
            torch.cuda.synchronize()
            energy = float(value.energy_eV.detach().cpu())
            force = np.asarray(
                value.force_source_order_eV_A.detach().cpu(), dtype=np.float64
            ).copy()
            if not math.isfinite(energy) or not np.all(np.isfinite(force)):
                raise FloatingPointError("non-finite O3 smoke E/F")
            states.append(r2o.state_dict_sha256(model))
            default_dtypes.append(str(torch.get_default_dtype()))
            return energy, force, value.query_receipt

        base_energy, base_force, base_query = combined(probe, reference)
        proper, improper = orthogonal_pair()
        o3 = {}
        arrays = {
            "baseline_energy_eV": np.asarray([base_energy], dtype=np.float64),
            "baseline_force_eV_A": base_force,
        }
        adapted_base = r2o.adapt_reference_cell(reference, reference)
        graph_dtype = r2o.model_dtype(model)
        base_graph = r2o.fixed_reference_graph(
            adapted_base, device="cuda", dtype=graph_dtype
        )
        base_graph_hash = r2r.tensor_mapping_semantic_sha256(
            base_graph, r2r.FORMAL_GRAPH_TENSOR_KEYS
        )
        if base_graph_hash != base_query["formal_R2O_graph_semantic_sha256"]:
            raise ValueError("independent baseline graph hash differs from public query")

        thresholds = r2r.CANONICAL_CONTRACT["fixed_gates"][
            "O3_proper_and_improper"
        ]
        for name, matrix in (("proper", proper), ("improper", improper)):
            transformed_reference = transformed_atoms(reference, matrix)
            transformed_probe = transformed_atoms(probe, matrix)
            transformed_reference_hash = r2r.validate_reference_semantics(
                transformed_reference
            )
            energy, force, query = combined(
                transformed_probe,
                transformed_reference,
                graph_mode="rigid_transform_probe",
                baseline_reference_template=reference,
                baseline_structure_template=probe,
                rigid_transform=matrix,
            )
            adapted_transformed = r2o.adapt_reference_cell(
                transformed_reference, transformed_probe
            )
            native_rebuilt_graph = r2o.fixed_reference_graph(
                adapted_transformed, device="cuda", dtype=graph_dtype
            )
            covariant_graph = r2r._covariant_formal_graph(base_graph, matrix)
            covariant_graph_hash = r2r.tensor_mapping_semantic_sha256(
                covariant_graph, r2r.FORMAL_GRAPH_TENSOR_KEYS
            )
            native_rebuilt_graph_hash = r2r.tensor_mapping_semantic_sha256(
                native_rebuilt_graph, r2r.FORMAL_GRAPH_TENSOR_KEYS
            )
            if covariant_graph_hash != query["formal_R2O_graph_semantic_sha256"]:
                raise ValueError(
                    f"independent {name} covariant graph differs from public query"
                )
            rigid_receipt = query["rigid_transform_receipt"]
            if (
                rigid_receipt["selected_graph_sha256"] != covariant_graph_hash
                or rigid_receipt["derived_graph_sha256"] != covariant_graph_hash
                or rigid_receipt["baseline_graph_sha256"] != base_graph_hash
                or rigid_receipt["native_rebuild_used_for_physics"] is not False
                or query["diagnostic_native_rebuild_selected"] is not False
                or rigid_receipt["native_rebuild_diagnostic"][
                    "physics_gate_authorized"
                ]
                is not False
            ):
                raise ValueError(f"{name} public query selected a non-covariant graph")
            energy_difference = abs(energy - base_energy)
            force_difference = float(
                np.max(np.abs(force - base_force @ matrix.T))
            )
            o3[name] = {
                "matrix": matrix.tolist(),
                "matrix_semantic_sha256": r2r.semantic_sha256(matrix.tolist()),
                "determinant": float(np.linalg.det(matrix)),
                "reference_semantic_sha256": transformed_reference_hash,
                "energy_eV": energy,
                "energy_abs_difference_eV": energy_difference,
                "force_covariance_max_abs_difference_eV_A": force_difference,
                "force_semantic_sha256": hashlib.sha256(
                    np.ascontiguousarray(force, dtype=np.float64).tobytes()
                ).hexdigest(),
                "query_receipt": query,
                "independent_covariant_graph_sha256": covariant_graph_hash,
                "independent_native_rebuilt_graph_sha256": native_rebuilt_graph_hash,
                "graph_geometry": graph_covariance_receipt(
                    base_graph, covariant_graph, native_rebuilt_graph, matrix
                ),
                "gate_pass": bool(
                    energy_difference <= thresholds["energy_abs_eV"]
                    and force_difference
                    <= thresholds["force_covariance_max_abs_eV_A"]
                ),
            }
            arrays[f"{name}_matrix"] = matrix
            arrays[f"{name}_energy_eV"] = np.asarray([energy], dtype=np.float64)
            arrays[f"{name}_force_eV_A"] = force

        if call_count != 3:
            raise ValueError("bounded smoke must make exactly three combined public calls")
        state_after = r2o.state_dict_sha256(model)
        default_dtype_after = str(torch.get_default_dtype())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        array_path = output / "o3_ef_arrays.npz"
        array_tmp = output / "o3_ef_arrays.npz.tmp"
        with array_tmp.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(array_tmp, array_path)
        array_schema = {
            key: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for key, value in arrays.items()
        }
        all_gate_pass = all(item["gate_pass"] for item in o3.values())
        state_unchanged = all(value == state_before for value in states + [state_after])
        dtype_unchanged = all(
            value == "torch.float32"
            for value in default_dtypes + [default_dtype_after]
        )
        receipt = {
            "format": FORMAT,
            "status": "O3_DIAGNOSTIC_PASS" if all_gate_pass else "O3_DIAGNOSTIC_FAIL",
            "diagnostic_only": True,
            "formal_GO_used": False,
            "can_authorize_fit_or_training": False,
            "shard_execution": False,
            "mechanics_execution": False,
            "full_formal_execution": False,
            "held_data_access": False,
            "support_data_access": False,
            "force_labels_used": False,
            "seed": 83,
            "thermal_global_index": 0,
            "thermal_config_type": probe.info["config_type"],
            "thermal_frame_count_geometry_only": len(thermal),
            "attempt2_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "manifest": manifest,
            "staged_source_sha256": source_hashes,
            "staged_input_sha256": input_hashes,
            "driver_sha256": sha256(driver_path),
            "combined_public_path": (
                "graphene_r2r_multipolar_background.production_combined_energy_force"
            ),
            "combined_public_call_count": call_count,
            "monkeypatch_or_override_used": False,
            "fixed_parameter_decomposition": {
                "executed": False,
                "reason": (
                    "bounded diagnostic uses only the three decisive combined public "
                    "E/F calls; no additional fixed/parameter production paths"
                ),
            },
            "endpoint_state_sha256_before": state_before,
            "endpoint_state_sha256_after": state_after,
            "endpoint_state_sha256_after_each_call": states[1:],
            "endpoint_state_unchanged": state_unchanged,
            "torch_default_dtype_before": default_dtype_before,
            "torch_default_dtype_after": default_dtype_after,
            "torch_default_dtype_after_each_call": default_dtypes[1:],
            "torch_default_dtype_unchanged_float32": dtype_unchanged,
            "reference_semantic_sha256": base_reference_hash,
            "baseline": {
                "energy_eV": base_energy,
                "force_semantic_sha256": hashlib.sha256(
                    np.ascontiguousarray(base_force, dtype=np.float64).tobytes()
                ).hexdigest(),
                "query_receipt": base_query,
                "independent_graph_sha256": base_graph_hash,
            },
            "O3": o3,
            "thresholds": thresholds,
            "arrays": {
                "basename": array_path.name,
                "sha256": sha256(array_path),
                "schema": array_schema,
            },
            "runtime": {
                "hostname": platform.node(),
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "elapsed_seconds": elapsed,
            },
        }
        receipt_path = output / "receipt.json"
        atomic_json(receipt_path, receipt)
        receipt_hash = sha256(receipt_path)
        if not all_gate_pass or not state_unchanged or not dtype_unchanged:
            atomic_json(
                output / "FAILED",
                {
                    "format": FORMAT,
                    "status": receipt["status"],
                    "receipt_sha256": receipt_hash,
                },
            )
            running.unlink()
            raise SystemExit(1)
        atomic_text(output / "DONE", "O3_DIAGNOSTIC_PASS\n" + receipt_hash + "\n")
        running.unlink()
    except BaseException as exception:
        if output.exists() and not (output / "DONE").exists() and not (output / "FAILED").exists():
            atomic_json(
                output / "FAILED",
                {
                    "format": FORMAT,
                    "status": "O3_DIAGNOSTIC_EXCEPTION",
                    "exception_type": type(exception).__name__,
                    "message": str(exception),
                    "can_authorize_fit_or_training": False,
                },
            )
        if running.exists():
            running.unlink()
        raise


if __name__ == "__main__":
    main()
