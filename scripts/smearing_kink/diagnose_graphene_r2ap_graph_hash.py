#!/usr/bin/env python3
"""Report the environment-dependent frozen R2O reference-graph fingerprint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
from ase.io import read


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for item in (HERE, ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import graphene_r2o_taylor_null as r2o  # noqa: E402
import graphene_r2r_multipolar_background as r2r  # noqa: E402
from graphene_r2r0_formal import load_endpoint  # noqa: E402
from train_graphene_r2s_conditional_mlp import recommended_inputs  # noqa: E402


def array_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--r2ap-manifest", type=Path)
    parser.add_argument("--lattice-temperature", type=int, default=450)
    args = parser.parse_args()

    inputs = recommended_inputs()
    model = load_endpoint(inputs, args.device)
    reference_template = read(inputs.reference_6x6, index=0)
    structure = reference_template
    context_record = {"kind": "raw_reference_template"}
    if args.r2ap_manifest is not None:
        import friedel_module as fm
        import run_graphene_physical_q0_sscha as q0
        import run_graphene_r2ap_fixed_smearing_sscha as r2ap

        manifest = json.loads(args.r2ap_manifest.resolve(strict=True).read_text())
        phonon = fm.load_ph(r2ap.resolved_record(manifest["background"]))
        initial_fc2, _ = r2ap.initial_force_constants(
            manifest, args.lattice_temperature
        )
        _, _, _, _, structure = q0.phonopy_fc_to_cc_dyn(phonon, initial_fc2)
        context_record = {
            "kind": "R2AP_CellConstructor_reference",
            "lattice_temperature_K": args.lattice_temperature,
            "cell_minus_template_max_abs_A": float(
                abs(
                    structure.cell.array
                    - reference_template.cell.array
                ).max()
            ),
        }
    reference = r2o.adapt_reference_cell(reference_template, structure)
    data = r2o.fixed_reference_graph(
        reference, device=args.device, dtype=r2o.model_dtype(model)
    )
    versions = {}
    for name in ("torch", "torch-geometric", "mace-torch", "ase"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    observed = r2r.tensor_mapping_semantic_sha256(
        data, r2r.FORMAL_GRAPH_TENSOR_KEYS
    )
    print(
        json.dumps(
            {
                "versions": versions,
                "device": args.device,
                "torch_default_dtype": str(torch.get_default_dtype()),
                "model_dtype": str(r2o.model_dtype(model)),
                "reference_file": str(inputs.reference_6x6),
                "reference_file_sha256": file_sha256(inputs.reference_6x6),
                "reference_semantic_sha256": r2r.reference_semantic_sha256(reference),
                "context": context_record,
                "adapted_cell_A": reference.cell.array.tolist(),
                "observed_graph_sha256": observed,
                "expected_graph_sha256": r2r.EXPECTED_BASELINE_FORMAL_GRAPH_SHA256[
                    "reference_6x6"
                ],
                "matches": observed
                == r2r.EXPECTED_BASELINE_FORMAL_GRAPH_SHA256["reference_6x6"],
                "tensor_records": {
                    key: {
                        "dtype": str(data[key].dtype),
                        "shape": list(data[key].shape),
                        "raw_sha256": array_sha256(data[key]),
                    }
                    for key in r2r.FORMAL_GRAPH_TENSOR_KEYS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
