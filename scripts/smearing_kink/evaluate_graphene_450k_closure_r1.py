"""Evaluate the frozen R1' overlap gates for the 450 K matched closure labels.

Compares the returned fixed-smearing DFT labels against
  * the stored R2AO forces/energies on the same bit-exact configs
    (order-safe paired reference, ``R2AP_fixed_smearing_SSCHA``), and
  * the matched-pool converged SSCHA Hessian (harmonic prediction).

Gates (frozen in ``R1_450k_matched_overlap_screen/freeze_manifest.json``
before any DFT was read):
  1. force component RMSE <= 30 meV/A, max <= 200 meV/A   (DFT vs R2AO)
  2. A'-projected force-difference RMS <= 15 meV/A
  3. A'-restoring slope ratio slope_R2AO/slope_DFT in [0.9, 1.1]
  4. A'-restoring slope relative error vs the SSCHA Hessian <= 5%
S0-specific force gates are deferred (delta checkpoint on the offline 2060).

Usage:
  python evaluate_graphene_450k_closure_r1.py \
      --shard-a-labels DIR --shard-b-labels DIR
"""
from __future__ import annotations

import argparse
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

from audit_graphene_current_s0_fixed_smearing import (  # noqa: E402
    folded_k_aprime_mode,
    load_operator,
)

BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
R2R = BASE / "R2R_multipolar_background"
SHARED = R2R / "R2AT_s0_shared_initial_sensitivity/formal_T450_Tel300"
OPERATOR = ROOT / "data/graphene_r2c_eval/operators/T300_operator.npz"
BACKGROUND = ROOT / "results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml"
OUT = BASE / "R1_450k_matched_overlap_screen"

MASS_C_AMU = 12.011
KB_EV_PER_K = 8.617333262e-5
RY_TO_EV = 13.605691932782346


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(directory: Path) -> dict[int, dict]:
    labels = {}
    for path in sorted(directory.glob("snapshot_*_k8.npz")):
        index = int(path.stem.split("_")[1])
        with np.load(path, allow_pickle=False) as data:
            labels[index] = {
                "positions": np.asarray(data["positions"], float),
                "forces": np.asarray(data["forces"], float),
                "energy": float(data["energy"]),
                "path": str(path),
            }
    if not labels:
        raise ValueError(f"no snapshot_*_k8.npz labels under {directory}")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-a-labels", type=Path, required=True)
    parser.add_argument("--shard-b-labels", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((OUT / "freeze_manifest.json").read_text())
    for shard in ("A", "B"):
        recorded = manifest["outputs"][f"shard_{shard}_snapshots_sha256"]
        if sha256(OUT / f"shard_{shard}_snapshots.npz") != recorded:
            raise ValueError(f"shard {shard} snapshot sha256 drift")

    snapshots = {}
    for shard in ("A", "B"):
        with np.load(OUT / f"shard_{shard}_snapshots.npz", allow_pickle=False) as data:
            snapshots[shard] = {key: np.asarray(data[key]) for key in data.files}

    dft = {}
    for shard, directory in (("A", args.shard_a_labels), ("B", args.shard_b_labels)):
        labels = load_labels(directory)
        sscha_indices = snapshots[shard]["sscha_indices"]
        for local_index, sscha_index in enumerate(sscha_indices):
            if local_index not in labels:
                raise ValueError(f"shard {shard} missing local index {local_index}")
            record = labels[local_index]
            if not np.allclose(record["positions"], snapshots[shard]["positions"][local_index], atol=1e-8):
                raise ValueError(f"shard {shard} label positions differ from snapshot")
            dft[int(sscha_index)] = {
                **record,
                "shard": shard,
                "selection_group": str(snapshots[shard]["selection_groups"][local_index]),
                "R2AO_forces": snapshots[shard]["R2AO_forces_eV_A"][local_index],
                "R2AO_energy": float(snapshots[shard]["R2AO_energies_eV"][local_index]),
                "positions": snapshots[shard]["positions"][local_index],
            }
    if len(dft) != 12:
        raise ValueError(f"expected 12 labels, found {len(dft)}")

    _, reference, cell, _ = load_operator(OPERATOR)
    mode, mode_provenance = folded_k_aprime_mode(
        BACKGROUND, SHARED / "result.npz", reference, cell
    )
    pattern = mode.reshape(-1).real
    pattern_norm = float(pattern @ pattern)

    with np.load(SHARED / "result.npz", allow_pickle=False) as data:
        fc2 = np.asarray(data["free_energy_fc2_eV_A2"], float)

    table = []
    for sscha_index, record in sorted(dft.items()):
        positions = record["positions"]
        delta = positions - reference
        fractional = delta @ np.linalg.inv(cell)
        fractional -= np.round(fractional)
        displacement = fractional @ cell

        forces_dft = record["forces"]
        forces_r2ao = record["R2AO_forces"]
        forces_phi = -np.einsum("ijab,jb->ia", fc2, displacement)

        q_coordinate = float(pattern @ displacement.reshape(-1)) / pattern_norm
        row = {
            "sscha_index": sscha_index,
            "shard": record["shard"],
            "selection_group": record["selection_group"],
            "Q_Aprime_A": q_coordinate,
            "F_Aprime_DFT_eV_A": float(pattern @ forces_dft.reshape(-1)) / pattern_norm,
            "F_Aprime_R2AO_eV_A": float(pattern @ forces_r2ao.reshape(-1)) / pattern_norm,
            "F_Aprime_PHI_eV_A": float(pattern @ forces_phi.reshape(-1)) / pattern_norm,
            "DFT_force_RMS_eV_A": float(np.sqrt((forces_dft**2).mean())),
            "PHI_vs_DFT_RMSE_meV_A": float(
                np.sqrt(((forces_phi - forces_dft) ** 2).mean()) * 1000
            ),
            "energy_DFT_eV": record["energy"],
            "energy_R2AO_eV": record["R2AO_energy"],
        }
        table.append(row)

    q = np.array([row["Q_Aprime_A"] for row in table])
    slope_dft = float(np.polyfit(q, [r["F_Aprime_DFT_eV_A"] for r in table], 1)[0])
    slope_r2ao = float(np.polyfit(q, [r["F_Aprime_R2AO_eV_A"] for r in table], 1)[0])
    slope_phi = float(np.polyfit(q, [r["F_Aprime_PHI_eV_A"] for r in table], 1)[0])

    force_delta = np.array(
        [dft[i]["forces"] - dft[i]["R2AO_forces"] for i in sorted(dft)]
    )
    aprime_delta = np.abs(
        np.array([mode.reshape(-1).conj() @ (dft[i]["forces"] - dft[i]["R2AO_forces"]).reshape(-1)
                  for i in sorted(dft)])
    )

    energy_dft = np.array([row["energy_DFT_eV"] for row in table])
    energy_r2ao = np.array([row["energy_R2AO_eV"] for row in table])
    energy_delta_centered = energy_r2ao - energy_dft
    energy_delta_centered -= energy_delta_centered.mean()
    weights = np.exp(
        -(energy_dft - energy_r2ao + (energy_dft - energy_r2ao).mean())
        / (KB_EV_PER_K * 450.0)
    )
    ess = float(weights.sum() ** 2 / (weights**2).sum() / len(weights))

    metrics = {
        "force_RMSE_meV_A": float(np.sqrt((force_delta**2).mean()) * 1000),
        "force_max_abs_meV_A": float(np.abs(force_delta).max() * 1000),
        "Aprime_diff_RMS_meV_A": float(np.sqrt((aprime_delta**2).mean()) * 1000),
        "Aprime_slope_DFT_eV_A2": slope_dft,
        "Aprime_slope_R2AO_eV_A2": slope_r2ao,
        "Aprime_slope_PHI_eV_A2": slope_phi,
        "Aprime_slope_ratio_R2AO_over_DFT": slope_r2ao / slope_dft,
        "Aprime_slope_rel_error_vs_PHI": abs(slope_dft - slope_phi) / abs(slope_phi),
        "DFT_force_RMS_eV_A": float(np.sqrt(np.mean([r["DFT_force_RMS_eV_A"] ** 2 for r in table]))),
        "PHI_vs_DFT_RMSE_meV_A": float(np.sqrt(np.mean([r["PHI_vs_DFT_RMSE_meV_A"] ** 2 for r in table]))),
        "R2AO_minus_DFT_centered_energy_RMSE_meV_config": float(
            np.sqrt((energy_delta_centered**2).mean()) * 1000
        ),
        "importance_weight_ESS_fraction_R2AO_vs_DFT": ess,
        "Aprime_mode": {
            "primitive_frequency_cm_1": mode_provenance["primitive_frequency_cm_1"],
            "primitive_mode_index": mode_provenance["primitive_mode_index"],
        },
    }
    gates = {
        "force_RMSE_le_30_meV_A": metrics["force_RMSE_meV_A"] <= 30.0,
        "force_max_le_200_meV_A": metrics["force_max_abs_meV_A"] <= 200.0,
        "Aprime_diff_RMS_le_15_meV_A": metrics["Aprime_diff_RMS_meV_A"] <= 15.0,
        "Aprime_slope_ratio_in_0.9_1.1": 0.9 <= metrics["Aprime_slope_ratio_R2AO_over_DFT"] <= 1.1,
        "Aprime_slope_rel_error_le_5pct": metrics["Aprime_slope_rel_error_vs_PHI"] <= 0.05,
    }

    payload = {
        "status": "all_gates_passed" if all(gates.values()) else "gate_failure",
        "scope": "R1' matched overlap screen (R2AO reference; S0 force gates deferred to offline 2060)",
        "n_labels": len(dft),
        "metrics": metrics,
        "gates": gates,
        "per_config": table,
        "inputs": {
            "shard_a_labels_dir": str(args.shard_a_labels),
            "shard_b_labels_dir": str(args.shard_b_labels),
            "manifest": str(OUT / "freeze_manifest.json"),
        },
    }
    output = OUT / "r1_evaluation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "gates": gates}, indent=2))
    print(f"wrote {output}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
