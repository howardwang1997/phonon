from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "smearing_kink"
sys.path.insert(0, str(SCRIPTS))

import graphene_r2o_tail_smoke as smoke  # noqa: E402
from graphene_r2o_routed_tail import (  # noqa: E402
    build_order_mic_query,
    load_formal_r2o_bundle,
    query_raw_invariants,
    r2o_invariant_schema,
    sha256,
    state_dict_sha256,
)
from graphene_r2o_tail_smoke import (  # noqa: E402
    FORMAL_ACTUAL_MODE,
    LAUNCH_PLAN_STATUS,
    NODE_RESULT_STATUS,
    SYNTHETIC_MODE,
    build_dual_node_launch_plan,
    evaluate_dual_node_results,
    freeze_runtime_contract,
    freeze_smoke_input_manifest,
    load_smoke_input,
    remote_launch,
    run_node_smoke,
    runtime_fingerprint,
    validate_dual_result,
    validate_dual_node_launch_plan,
    validate_node_result,
    validate_formal_smoke_go_marker,
    validate_runtime_contract,
    write_tail_checkpoint,
)


def _load_base_helpers():
    path = Path(__file__).with_name("test_graphene_r2o_routed_tail.py")
    specification = importlib.util.spec_from_file_location(
        "_r2o_routed_tail_test_helpers", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


BASE = _load_base_helpers()


@pytest.fixture(scope="module")
def synthetic_smoke_tree(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("r2o_smoke_portable")
    hashes = BASE._write_formal_portable_tree(
        root, model=BASE._make_formal_shape_mace()
    )
    formal = load_formal_r2o_bundle(
        root,
        "formal_receipt.json",
        expected_receipt_sha256=hashes["receipt"],
        expected_gate_sha256=hashes["gate"],
        expected_bundle_sha256=hashes["bundle"],
        device="cpu",
    )
    schema = r2o_invariant_schema(formal)
    pristine_query = build_order_mic_query(formal, formal.reference_for_count(72))
    pristine = query_raw_invariants(formal, pristine_query, schema).detach()
    tail = BASE._random_tail(pristine)
    artifact_root = root / "smoke_artifacts"
    artifact_root.mkdir()
    tail_path = artifact_root / "actual_tail.pt"
    tail_hash = write_tail_checkpoint(tail_path, formal, tail, schema)
    runtime_path = artifact_root / "runtime_contract.json"
    freeze_runtime_contract(runtime_path, device="cpu")
    runtime_hash = sha256(runtime_path)
    manifest_path = artifact_root / "smoke_input.json"
    freeze_smoke_input_manifest(
        root,
        "smoke_artifacts/smoke_input.json",
        mode=SYNTHETIC_MODE,
        formal_receipt_relative_path="formal_receipt.json",
        expected_formal_receipt_sha256=hashes["receipt"],
        expected_gate_sha256=hashes["gate"],
        expected_bundle_sha256=hashes["bundle"],
        tail_checkpoint_relative_path="smoke_artifacts/actual_tail.pt",
        expected_tail_checkpoint_sha256=tail_hash,
        expected_tail_state_sha256=state_dict_sha256(tail),
        expected_schema_sha256=schema["schema_sha256"],
        runtime_contract_relative_path="smoke_artifacts/runtime_contract.json",
        expected_runtime_contract_sha256=runtime_hash,
    )
    return {
        "root": root,
        "hashes": hashes,
        "manifest_relative": "smoke_artifacts/smoke_input.json",
        "manifest_sha256": sha256(manifest_path),
        "tail_state_sha256": state_dict_sha256(tail),
        "tail_checkpoint_sha256": tail_hash,
        "schema_sha256": schema["schema_sha256"],
        "runtime_contract_sha256": runtime_hash,
    }


def test_runtime_fingerprint_is_canonical_and_formal_mode_requires_cuda(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.json"
    contract = freeze_runtime_contract(path, device="cpu")
    fingerprint = runtime_fingerprint("cpu")
    validated = validate_runtime_contract(
        path,
        expected_sha256=sha256(path),
        fingerprint=fingerprint,
        mode=SYNTHETIC_MODE,
    )
    assert validated["contract_sha256"] == contract["contract_sha256"]
    changed = copy.deepcopy(fingerprint)
    changed["semantic_environment"]["numpy_version"] = "0.0-tampered"
    changed["semantic_environment_sha256"] = smoke.canonical_json_sha256(
        changed["semantic_environment"]
    )
    with pytest.raises(ValueError, match="semantic environment differs"):
        validate_runtime_contract(
            path,
            expected_sha256=sha256(path),
            fingerprint=changed,
            mode=SYNTHETIC_MODE,
        )
    with pytest.raises(ValueError, match="requires CUDA"):
        validate_runtime_contract(
            path,
            expected_sha256=sha256(path),
            fingerprint=fingerprint,
            mode=FORMAL_ACTUAL_MODE,
        )


def test_smoke_input_loads_only_hash_bound_formal_and_nonzero_tail(
    synthetic_smoke_tree: dict,
) -> None:
    loaded = load_smoke_input(
        synthetic_smoke_tree["root"],
        synthetic_smoke_tree["manifest_relative"],
        expected_manifest_sha256=synthetic_smoke_tree["manifest_sha256"],
        device="cpu",
    )
    assert state_dict_sha256(loaded.tail) == synthetic_smoke_tree[
        "tail_state_sha256"
    ]
    assert all(parameter.dtype == torch.float64 for parameter in loaded.tail.parameters())
    assert all(not parameter.requires_grad for parameter in loaded.tail.parameters())
    assert loaded.manifest["forbidden"] == {
        "support9_opened": False,
        "seed2_opened": False,
        "outer_folds_prepared": False,
        "remote_launch_authorized": False,
    }


def test_manifest_rejects_unopened_data_paths_before_any_model_load(
    synthetic_smoke_tree: dict,
) -> None:
    root = synthetic_smoke_tree["root"]
    with pytest.raises(ValueError, match="unopened or reserved data"):
        freeze_smoke_input_manifest(
            root,
            "smoke_artifacts/bad_input.json",
            mode=SYNTHETIC_MODE,
            formal_receipt_relative_path="unopened_support/formal.json",
            expected_formal_receipt_sha256="00" * 32,
            expected_gate_sha256="00" * 32,
            expected_bundle_sha256="00" * 32,
            tail_checkpoint_relative_path="smoke_artifacts/actual_tail.pt",
            expected_tail_checkpoint_sha256="00" * 32,
            expected_tail_state_sha256="00" * 32,
            expected_schema_sha256="00" * 32,
            runtime_contract_relative_path="smoke_artifacts/runtime_contract.json",
            expected_runtime_contract_sha256="00" * 32,
        )


def test_nested_formal_receipt_paths_reject_forbidden_tokens_before_load(
    synthetic_smoke_tree: dict, tmp_path: Path
) -> None:
    copied = tmp_path / "portable_copy"
    shutil.copytree(synthetic_smoke_tree["root"], copied)
    receipt_path = copied / "formal_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["paths"]["gate"] = "unopened_support/core_gate.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    manifest_path = copied / synthetic_smoke_tree["manifest_relative"]
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"]["formal_receipt"] = sha256(receipt_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="unopened or reserved data"):
        load_smoke_input(
            copied,
            synthetic_smoke_tree["manifest_relative"],
            expected_manifest_sha256=sha256(manifest_path),
            device="cpu",
        )


def test_formal_mode_requires_independent_go_marker_before_runtime_or_model_load(
    synthetic_smoke_tree: dict, tmp_path: Path
) -> None:
    root = synthetic_smoke_tree["root"]
    formal_manifest = root / "smoke_artifacts" / "formal_input.json"
    freeze_smoke_input_manifest(
        root,
        "smoke_artifacts/formal_input.json",
        mode=FORMAL_ACTUAL_MODE,
        formal_receipt_relative_path="formal_receipt.json",
        expected_formal_receipt_sha256=synthetic_smoke_tree["hashes"]["receipt"],
        expected_gate_sha256=synthetic_smoke_tree["hashes"]["gate"],
        expected_bundle_sha256=synthetic_smoke_tree["hashes"]["bundle"],
        tail_checkpoint_relative_path="smoke_artifacts/actual_tail.pt",
        expected_tail_checkpoint_sha256=synthetic_smoke_tree[
            "tail_checkpoint_sha256"
        ],
        expected_tail_state_sha256=synthetic_smoke_tree["tail_state_sha256"],
        expected_schema_sha256=synthetic_smoke_tree["schema_sha256"],
        runtime_contract_relative_path="smoke_artifacts/runtime_contract.json",
        expected_runtime_contract_sha256=synthetic_smoke_tree[
            "runtime_contract_sha256"
        ],
    )
    formal_manifest_hash = sha256(formal_manifest)
    with pytest.raises(ValueError, match="requires an independent GO marker"):
        run_node_smoke(
            root,
            "smoke_artifacts/formal_input.json",
            expected_manifest_sha256=formal_manifest_hash,
            node_label="formal-a",
            output_root=tmp_path / "formal_missing_go",
            device="cpu",
        )
    marker = {
        "format": smoke.FORMAL_GO_FORMAT,
        "status": smoke.FORMAL_GO_STATUS,
        "scope": "real_formal_no_support_actual_tail_smoke_only",
        "input_manifest_sha256": formal_manifest_hash,
        "authorized_node_labels": ["formal-a", "formal-b"],
        "independent_review_receipt_sha256": "11" * 32,
        "source_sha256": smoke.smoke_source_hashes(),
        "protocol_sha256": smoke.smoke_protocol()["protocol_sha256"],
        "real_formal_smoke_authorized": True,
        "outer_prepare_authorized": False,
    }
    marker["authorization_sha256"] = smoke.canonical_json_sha256(marker)
    marker_path = root / "smoke_artifacts" / "review_go.json"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    validated = validate_formal_smoke_go_marker(
        root,
        "smoke_artifacts/review_go.json",
        expected_marker_sha256=sha256(marker_path),
        input_manifest_sha256=formal_manifest_hash,
        node_label="formal-a",
    )
    assert validated["outer_prepare_authorized"] is False
    with pytest.raises(ValueError, match="requires CUDA"):
        run_node_smoke(
            root,
            "smoke_artifacts/formal_input.json",
            expected_manifest_sha256=formal_manifest_hash,
            node_label="formal-a",
            output_root=tmp_path / "formal_with_go",
            device="cpu",
            formal_go_marker_relative_path="smoke_artifacts/review_go.json",
            expected_formal_go_marker_sha256=sha256(marker_path),
        )


@pytest.fixture(scope="module")
def completed_node_smokes(
    synthetic_smoke_tree: dict, tmp_path_factory: pytest.TempPathFactory
) -> dict:
    output = tmp_path_factory.mktemp("r2o_smoke_outputs")
    first = run_node_smoke(
        synthetic_smoke_tree["root"],
        synthetic_smoke_tree["manifest_relative"],
        expected_manifest_sha256=synthetic_smoke_tree["manifest_sha256"],
        node_label="synthetic-a",
        output_root=output / "node_a",
        device="cpu",
        synthetic_hostname_override="synthetic-host-a",
    )
    second = run_node_smoke(
        synthetic_smoke_tree["root"],
        synthetic_smoke_tree["manifest_relative"],
        expected_manifest_sha256=synthetic_smoke_tree["manifest_sha256"],
        node_label="synthetic-b",
        output_root=output / "node_b",
        device="cpu",
        synthetic_hostname_override="synthetic-host-b",
    )
    return {"output": output, "first": first, "second": second}


def test_actual_core_tail_smoke_covers_null_fd_o3_order_mic_and_graph_semantics(
    completed_node_smokes: dict,
) -> None:
    output = completed_node_smokes["output"]
    first = completed_node_smokes["first"]
    assert first["result"]["status"] == NODE_RESULT_STATUS
    assert first["recovered"] is False
    validated = validate_node_result(
        output / "node_a", expected_result_sha256=first["result_sha256"]
    )
    assert validated["passes_all_gates"] is True
    metrics = json.loads((output / "node_a" / "metrics.json").read_text())
    assert metrics["passes_all_gates"] is True
    assert all(metrics["gates"].values()), metrics["gates"]
    assert metrics["raw_MACE_direct_deployment"] is False
    assert metrics["actual_tail"]["tail_force_max_abs_eV_A_at_probe"] > 0.0
    assert metrics["reference_null"]["Hessian_semantic_sha256"]
    assert metrics["nonreference_complete_Hessian"]["shape"] == [216, 216]
    assert metrics["nonreference_complete_Hessian"][
        "antisymmetry_max_abs_eV_A2"
    ] <= 1.0e-7
    assert metrics["nonreference_complete_Hessian"][
        "translation_ASR_max_abs_eV_A2"
    ] <= 1.0e-7
    assert metrics["order_MIC"]["has_nonzero_MIC_integer"] is True
    assert metrics["paired_native_FP64_sensitivity"]["6x6_72_atom"][
        "passes_paired_native_FP64_sensitivity"
    ] is True
    assert metrics["paired_native_FP64_sensitivity"]["8x8_128_atom"][
        "passes_paired_native_FP64_sensitivity"
    ] is True
    assert metrics["paired_native_FP64_sensitivity"]["6x6_72_atom"][
        "energy_size_normalization"
    ]["atom_count"] == 72
    assert metrics["paired_native_FP64_sensitivity"]["8x8_128_atom"][
        "energy_size_normalization"
    ]["atom_count"] == 128
    assert metrics["size_6x6_8x8"]["energy_abs_difference_eV"] <= 1.0e-7
    assert metrics["size_6x6_8x8"][
        "central_force_max_abs_difference_eV_A"
    ] <= 1.0e-5
    assert metrics["graph_source_geometry"]["6x6_observed_max_A"] <= 1.0e-6
    assert 5.0e-7 < metrics["graph_source_geometry"][
        "8x8_observed_max_A"
    ] <= 1.0e-6
    assert set(metrics["O3"]) == {"proper", "improper"}


def test_completed_node_recovery_is_validation_only_and_tamper_fails_closed(
    synthetic_smoke_tree: dict, completed_node_smokes: dict
) -> None:
    output = completed_node_smokes["output"]
    first = completed_node_smokes["first"]
    recovered = run_node_smoke(
        synthetic_smoke_tree["root"],
        synthetic_smoke_tree["manifest_relative"],
        expected_manifest_sha256=synthetic_smoke_tree["manifest_sha256"],
        node_label="synthetic-a",
        output_root=output / "node_a",
        device="cpu",
        evaluator=lambda loaded: (_ for _ in ()).throw(
            AssertionError("completed recovery must not recompute")
        ),
    )
    assert recovered["recovered"] is True
    assert recovered["result_sha256"] == first["result_sha256"]
    metrics_path = output / "node_a" / "metrics.json"
    original = metrics_path.read_bytes()
    metrics_path.write_bytes(original + b"\n")
    try:
        with pytest.raises(ValueError, match="artifact changed"):
            validate_node_result(
                output / "node_a", expected_result_sha256=first["result_sha256"]
            )
    finally:
        metrics_path.write_bytes(original)


def test_failed_node_publishes_failure_marker_and_refuses_incomplete_reuse(
    synthetic_smoke_tree: dict, tmp_path: Path
) -> None:
    output = tmp_path / "failure_case"
    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        run_node_smoke(
            synthetic_smoke_tree["root"],
            synthetic_smoke_tree["manifest_relative"],
            expected_manifest_sha256=synthetic_smoke_tree["manifest_sha256"],
            node_label="failure-node",
            output_root=output,
            device="cpu",
            evaluator=lambda loaded: (_ for _ in ()).throw(
                RuntimeError("synthetic evaluator failure")
            ),
        )
    assert (output / "FAILED").is_file()
    assert (output / "EXIT_CODE").read_text() == "1\n"
    assert not (output / "RUNNING").exists()
    with pytest.raises(ValueError, match="refuses in-place recovery"):
        run_node_smoke(
            synthetic_smoke_tree["root"],
            synthetic_smoke_tree["manifest_relative"],
            expected_manifest_sha256=synthetic_smoke_tree["manifest_sha256"],
            node_label="failure-node",
            output_root=output,
            device="cpu",
        )


def test_dual_evaluator_requires_same_frozen_bindings_and_is_recoverable(
    synthetic_smoke_tree: dict, completed_node_smokes: dict,
) -> None:
    output = completed_node_smokes["output"]
    first = completed_node_smokes["first"]
    second = completed_node_smokes["second"]
    dual = evaluate_dual_node_results(
        [
            (output / "node_a", first["result_sha256"]),
            (output / "node_b", second["result_sha256"]),
        ],
        output_root=output / "dual",
    )
    assert dual["result"]["status"] == smoke.DUAL_RESULT_STATUS
    assert dual["result"]["remote_launch_or_outer_prepare_authorized"] is False
    validated = validate_dual_result(
        output / "dual", expected_result_sha256=dual["result_sha256"]
    )
    assert validated["passes_all_gates"] is True
    recovered = evaluate_dual_node_results(
        [
            (output / "node_a", first["result_sha256"]),
            (output / "node_b", second["result_sha256"]),
        ],
        output_root=output / "dual",
    )
    assert recovered["recovered"] is True
    assert recovered["result_sha256"] == dual["result_sha256"]

    first_metrics = json.loads((output / "node_a" / "metrics.json").read_text())
    same_host = run_node_smoke(
        synthetic_smoke_tree["root"],
        synthetic_smoke_tree["manifest_relative"],
        expected_manifest_sha256=synthetic_smoke_tree["manifest_sha256"],
        node_label="synthetic-c",
        output_root=output / "node_same_host",
        device="cpu",
        synthetic_hostname_override="synthetic-host-a",
        evaluator=lambda loaded: copy.deepcopy(first_metrics),
    )
    with pytest.raises(ValueError, match="distinct runtime hostnames"):
        evaluate_dual_node_results(
            [
                (output / "node_a", first["result_sha256"]),
                (output / "node_same_host", same_host["result_sha256"]),
            ],
            output_root=output / "dual_same_host",
        )


def test_launcher_is_tailscale_plan_only_and_cannot_execute(
    synthetic_smoke_tree: dict, tmp_path: Path
) -> None:
    nodes = [
        {
            "label": "v100-a",
            "target": "100.64.0.11",
            "transport": "tailscale_ssh",
            "remote_portable_root": "/opt/phonon/r2o_smoke",
            "remote_output_root": "/opt/phonon/r2o_smoke_runs/node_a",
            "device": "cuda:0",
        },
        {
            "label": "v100-b",
            "target": "100.64.0.12",
            "transport": "openssh_over_tailscale_tun",
            "remote_portable_root": "/srv/phonon/r2o_smoke",
            "remote_output_root": "/srv/phonon/r2o_smoke_runs/node_b",
            "device": "cuda:0",
        },
    ]
    plan_path = tmp_path / "launch_plan.json"
    plan = build_dual_node_launch_plan(
        synthetic_smoke_tree["root"],
        synthetic_smoke_tree["manifest_relative"],
        expected_manifest_sha256=synthetic_smoke_tree["manifest_sha256"],
        nodes=nodes,
        output_path=plan_path,
    )
    assert plan["status"] == LAUNCH_PLAN_STATUS
    assert plan["remote_execution_authorized"] is False
    assert plan["outer_prepare_authorized"] is False
    assert all(record["remote_execution_authorized"] is False for record in plan["nodes"])
    assert plan["nodes"][0]["argv"][:3] == [
        "tailscale",
        "ssh",
        "100.64.0.11",
    ]
    assert plan["nodes"][1]["argv"][:2] == ["ssh", "100.64.0.12"]
    validated = validate_dual_node_launch_plan(
        plan_path, expected_plan_artifact_sha256=sha256(plan_path)
    )
    assert validated["plan_sha256"] == plan["plan_sha256"]
    with pytest.raises(PermissionError, match="intentionally disabled"):
        remote_launch(plan)
    public_nodes = copy.deepcopy(nodes)
    public_nodes[0]["target"] = "8.8.8.8"
    with pytest.raises(ValueError, match="Tailscale"):
        build_dual_node_launch_plan(
            synthetic_smoke_tree["root"],
            synthetic_smoke_tree["manifest_relative"],
            expected_manifest_sha256=synthetic_smoke_tree["manifest_sha256"],
            nodes=public_nodes,
            output_path=tmp_path / "bad_plan.json",
        )
