#!/usr/bin/env python3
"""Local command line entry point for R2R-1 preflight, fit, and recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import aggregate_graphene_r2r1_linear_readout as aggregate
import graphene_r2r1_linear_readout as linear


DEFAULT_THERMAL92 = (
    linear.ROOT / "data/graphene_r2o_taylor_null_core/train_thermal.xyz"
)


def write_freeze_manifest(path: Path) -> dict[str, Any]:
    """Write only a candidate manifest; never create an authorization marker."""
    lexical = Path(os.path.abspath(str(path)))
    expected = Path(os.path.abspath(str(linear.RECOMMENDED_FREEZE_MANIFEST)))
    if lexical != expected:
        raise PermissionError("freeze manifest output differs from frozen production path")
    target = linear._lexical_path(
        path,
        "R2R-1 freeze manifest output",
        forbidden_tokens=linear.FORBIDDEN_LABEL_PATH_TOKENS,
    )
    control = linear._lexical_path(
        linear.RECOMMENDED_CONTROL_ROOT,
        "R2R-1 frozen control root",
        forbidden_tokens=linear.FORBIDDEN_LABEL_PATH_TOKENS,
    )
    if target.parent != control:
        raise PermissionError("R2R-1 freeze manifest escaped the control root")
    _parent, parent_fd, parent_chain = linear._open_directory_chain(
        control.parent, "R2R-1 frozen control parent"
    )
    del _parent
    control_fd: int | None = None
    try:
        try:
            before = os.stat(control.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            os.mkdir(control.name, 0o700, dir_fd=parent_fd)
            before = os.stat(control.name, dir_fd=parent_fd, follow_symlinks=False)
            os.fsync(parent_fd)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ValueError("R2R-1 control root is not a directory")
        control_fd = os.open(
            control.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        control_identity = linear._directory_binding_identity(os.fstat(control_fd))
        if control_identity != linear._directory_binding_identity(before):
            raise ValueError("R2R-1 control root changed across stat/open")
        control_chain = (*parent_chain, control_identity)
        names = set(os.listdir(control_fd))
        marker_name = linear.RECOMMENDED_AUTHORIZATION_MARKER.name
        if marker_name in names:
            raise FileExistsError(
                "remove the external GO marker before regenerating manifest"
            )
        unexpected = names - {target.name}
        if unexpected:
            raise ValueError(
                f"R2R-1 control root contains unexpected entries: {sorted(unexpected)}"
            )
        payload = linear.freeze_manifest_payload(
            control_root_binding={
                "path": str(control),
                "directory_identity": control_identity,
                "chain": [dict(item) for item in control_chain],
                "payload_bytes_read": 0,
            },
            held_result_parent_binding=linear._result_parent_binding_from_fd(
                parent=control.parent,
                descriptor=parent_fd,
                chain=parent_chain,
            ),
        )
        raw = linear.canonical_json_bytes(payload)
        if target.name in names:
            existing = aggregate._read_regular_file_at(
                control_fd,
                target.name,
                "existing R2R-1 freeze manifest",
                size_limit=8 * 1024 * 1024,
            )
            if existing != raw:
                raise FileExistsError(
                    "existing R2R-1 freeze manifest differs; use a fresh control root"
                )
        else:
            aggregate._atomic_write_bytes_at(control_fd, target.name, raw)
        manifest_identity = linear._owned_regular_identity(
            os.stat(target.name, dir_fd=control_fd, follow_symlinks=False)
        )
        initial_raw, observed_manifest_identity = (
            linear._read_owned_regular_file_at_fd(
                control_fd,
                target.name,
                "R2R-1 generated freeze manifest",
                size_limit=8 * 1024 * 1024,
                expected_identity=manifest_identity,
            )
        )
        if initial_raw != raw or not linear._json_type_exact_equal(
            observed_manifest_identity, manifest_identity
        ):
            raise ValueError("R2R-1 generated freeze manifest changed after write")
        final_names = set(os.listdir(control_fd))
        if final_names != {target.name} or marker_name in final_names:
            raise RuntimeError("manifest generator changed the exact control inventory")
        os.fsync(control_fd)
        linear._verify_directory_chain(
            control, "R2R-1 control root final binding", control_chain
        )
        final_raw, final_manifest_identity = linear._read_owned_regular_file_at_fd(
            control_fd,
            target.name,
            "R2R-1 freeze manifest after final control rebind",
            size_limit=8 * 1024 * 1024,
            expected_identity=manifest_identity,
        )
        if (
            final_raw != raw
            or hashlib.sha256(final_raw).digest() != hashlib.sha256(raw).digest()
            or not linear._json_type_exact_equal(
                final_manifest_identity, manifest_identity
            )
            or set(os.listdir(control_fd)) != {target.name}
            or not linear._json_type_exact_equal(
                linear._directory_binding_identity(os.fstat(control_fd)),
                control_identity,
            )
        ):
            raise ValueError("R2R-1 freeze manifest changed at generator return")
    finally:
        if control_fd is not None:
            os.close(control_fd)
        os.close(parent_fd)
    return {
        "format": "graphene_r2r1_manifest_generation_receipt_v1",
        "manifest_path": str(target),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "authorization_marker_created": False,
        "fit_performed": False,
        "thermal_force_labels_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--attempt3-root", type=Path, default=linear.ATTEMPT3_ROOT)
    preflight.add_argument("--thermal92", type=Path, default=DEFAULT_THERMAL92)

    for command in ("fit", "anchor-completed"):
        fit = subparsers.add_parser(command)
        fit.add_argument("--output-root", type=Path, required=True)
        fit.add_argument("--attempt3-root", type=Path, default=linear.ATTEMPT3_ROOT)
        fit.add_argument("--thermal92", type=Path, default=DEFAULT_THERMAL92)
        fit.add_argument("--freeze-manifest", type=Path, required=True)
        fit.add_argument("--authorization-marker", type=Path, required=True)
        fit.add_argument("--release-manifest", type=Path, required=True)

    recover = subparsers.add_parser("recover")
    recover.add_argument("--output-root", type=Path, required=True)
    recover.add_argument("--attempt3-root", type=Path, default=linear.ATTEMPT3_ROOT)
    recover.add_argument("--thermal92", type=Path, default=DEFAULT_THERMAL92)
    recover.add_argument("--freeze-manifest", type=Path, required=True)
    recover.add_argument("--authorization-marker", type=Path, required=True)
    recover.add_argument("--release-manifest", type=Path, required=True)
    recover.add_argument("--expected-release-manifest-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "manifest":
        receipt = write_freeze_manifest(args.output)
    elif args.command == "preflight":
        receipt = linear.label_blind_preflight(args.attempt3_root, args.thermal92)
    elif args.command == "fit":
        aggregate._validate_production_path_contract(
            output_root=args.output_root,
            attempt3_root=args.attempt3_root,
            thermal92_path=args.thermal92,
            freeze_manifest=args.freeze_manifest,
            authorization_marker=args.authorization_marker,
            release_manifest=args.release_manifest,
            input_existence_required=False,
        )
        receipt = aggregate.materialize_authorized_fit(
            output_root=args.output_root,
            attempt3_root=args.attempt3_root,
            thermal92_path=args.thermal92,
            freeze_manifest=args.freeze_manifest,
            authorization_marker=args.authorization_marker,
        )
        receipt = {
            **receipt,
            "external_release_manifest": aggregate.anchor_completed_release(
                output_root=args.output_root,
                attempt3_root=args.attempt3_root,
                thermal92_path=args.thermal92,
                freeze_manifest=args.freeze_manifest,
                authorization_marker=args.authorization_marker,
                release_manifest=args.release_manifest,
                materialization_witness=receipt.anchor_witness,
            ),
        }
    elif args.command == "anchor-completed":
        receipt = aggregate.anchor_completed_release(
            output_root=args.output_root,
            attempt3_root=args.attempt3_root,
            thermal92_path=args.thermal92,
            freeze_manifest=args.freeze_manifest,
            authorization_marker=args.authorization_marker,
            release_manifest=args.release_manifest,
        )
    else:
        receipt = aggregate.recover_completed_fit(
            output_root=args.output_root,
            attempt3_root=args.attempt3_root,
            thermal92_path=args.thermal92,
            freeze_manifest=args.freeze_manifest,
            authorization_marker=args.authorization_marker,
            release_manifest=args.release_manifest,
            expected_release_manifest_sha256=(
                args.expected_release_manifest_sha256
            ),
        )
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
