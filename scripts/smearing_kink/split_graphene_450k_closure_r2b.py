"""Split the frozen R2B tagged-pair snapshots into two execution halves.

Pure execution aid: the frozen artifact is ``r2b_tagged_pairs_snapshots.npz``
(24 configs); each box runs one 12-config half through the standard batch
runner.  ``sscha_indices`` keep the frozen label ids (1000 + 2*i + sign) so
the pulled labels can be reassembled into the canonical
``labels/r2b_tagged_pairs/snapshot_{local:03d}_k8.npz`` tree, where
``local`` indexes the FULL frozen npz (local = label_id - 1000).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R1_450k_matched_overlap_screen"
)
FULL = OUT / "r2b_tagged_pairs_snapshots.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    with np.load(FULL, allow_pickle=False) as data:
        full = {key: np.asarray(data[key]) for key in data.files}
    n = len(full["positions"])
    if n != 24:
        raise ValueError(f"expected 24 tagged configs, found {n}")
    outputs = {}
    for name, slice_ in (("r2b_half1", slice(0, 12)), ("r2b_half2", slice(12, 24))):
        payload = {
            key: (value[slice_] if value.ndim >= 1 and value.shape[0] == n else value)
            for key, value in full.items()
        }
        path = OUT / f"{name}_snapshots.npz"
        np.savez(path, **payload)
        with np.load(path, allow_pickle=False) as check:
            if not np.array_equal(np.asarray(check["sscha_indices"]), full["sscha_indices"][slice_]):
                raise ValueError(f"{name} label ids drift")
            if not np.array_equal(np.asarray(check["positions"]), full["positions"][slice_]):
                raise ValueError(f"{name} positions drift")
        outputs[f"{name}_snapshots_sha256"] = sha256(path)
        print(f"wrote {path}")

    record = {
        "status": "split_from_frozen_r2b",
        "frozen": str(FULL),
        "frozen_sha256": sha256(FULL),
        "label_id_note": "sscha_indices keep frozen ids; canonical local index = id - 1000",
        "outputs": outputs,
    }
    (OUT / "r2b_split_manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT / 'r2b_split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
