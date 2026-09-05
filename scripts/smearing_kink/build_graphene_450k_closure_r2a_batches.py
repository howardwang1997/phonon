"""Build the reserved R2A batch snapshot files for the 450 K closure.

Reads the frozen reserve from ``R1_450k_matched_overlap_screen/freeze_manifest.json``
(batches 2-4, 12 configs each) and exports snapshot npz files in the same
format as the R1' shards (phonopy atom order, with frozen R2AO reference
forces/energies).  Selection was frozen before any DFT was submitted.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from audit_graphene_current_s0_fixed_smearing import load_operator  # noqa: E402

BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
R2R = BASE / "R2R_multipolar_background"
SHARED = R2R / "R2AT_s0_shared_initial_sensitivity/formal_T450_Tel300"
PAIRED = R2R / "R2AP_fixed_smearing_SSCHA/formal_T450"
OUT = BASE / "R1_450k_matched_overlap_screen"
RY_TO_EV = 13.605691932782346


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads((OUT / "freeze_manifest.json").read_text())
    acceptance = json.loads((SHARED / "acceptance.json").read_text())

    xats = np.load(SHARED / "ensembles/xats_pop1.npy")
    forces = np.load(PAIRED / "ensembles/forces_pop1.npy")
    energies = np.load(PAIRED / "ensembles/energies_pop1.npy")
    if not np.array_equal(xats, np.load(PAIRED / "ensembles/xats_pop1.npy")):
        raise ValueError("paired ensembles drifted")

    cc_to_phonopy = np.asarray(
        acceptance["atom_order_interface"]["CellConstructor_for_phonopy"], int
    )
    _, reference, cell, _ = load_operator(
        ROOT / "data/graphene_r2c_eval/operators/T300_operator.npz"
    )
    xats = xats[:, cc_to_phonopy, :]
    forces = forces[:, cc_to_phonopy, :]

    reserved = manifest["selection_protocol"]["reserved_R2A_batches"]
    already = sorted(
        index for batch in manifest["shards"].values() for index in batch
    )
    seen = set(already)
    outputs = {}
    for name, indices in reserved.items():
        indices = sorted(int(i) for i in indices)
        if len(indices) != 12 or len(set(indices)) != 12:
            raise ValueError(f"{name} is not 12 distinct indices")
        for index in indices:
            if index in seen:
                raise ValueError(f"index {index} appears twice across batches")
            seen.add(index)
        np.savez(
            OUT / f"{name}_snapshots.npz",
            positions=xats[indices],
            cells=np.repeat(cell[None, :, :], 12, axis=0),
            numbers=np.full(72, 6, dtype=int),
            sscha_indices=np.array(indices, dtype=int),
            selection_groups=np.array(["r2a_reserve"] * 12, dtype="<U16"),
            R2AO_forces_eV_A=forces[indices] * RY_TO_EV,
            R2AO_energies_eV=energies[indices] * RY_TO_EV,
        )
        path = OUT / f"{name}_snapshots.npz"
        outputs[f"{name}_snapshots_sha256"] = sha256(path)
        print(f"wrote {path}")

    record = OUT / "r2a_batch_manifest.json"
    record.write_text(
        json.dumps(
            {
                "status": "built_from_frozen_reserve",
                "source_manifest": str(OUT / "freeze_manifest.json"),
                "outputs": outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
