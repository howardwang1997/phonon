#!/usr/bin/env python3
"""Test whether the complete 512-column bilinear space can pass seed012 OOF."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_graphene_r2z_forward_bilinear as r2z
import fit_graphene_r2ad_seed012_step32 as r2ad
import graphene_r2r1_linear_readout as r2r1
from diagnose_graphene_r2r1_aprime_objective import canonical_json_bytes, file_sha256
from evaluate_graphene_r2t_bilinear_nested_objective import design_audit


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "results/graphene_physics_temperature/post_p4_feasibility"
    / "R2R_multipolar_background"
)
DEFAULT_OUTPUT = BASE / "R2AF_seed012_full512_development_20260826"
ALPHAS = tuple(float(value) for value in np.geomspace(3.0e-3, 3.0e-1, 25))


def candidate_systems() -> tuple[dict[str, Any], ...]:
    records = []
    default_weights = (0.45, 0.45, 0.10)
    for mass in (0.40, 0.55, 0.70):
        for penalty in (2.0, 5.0, 10.0):
            records.append(
                {
                    "projection_mass": mass,
                    "seed_weights": default_weights,
                    "bilinear_penalty": penalty,
                    "T600_force_fraction": 0.97,
                }
            )
    records.extend(
        (
            {
                "projection_mass": 0.55,
                "seed_weights": (0.40, 0.40, 0.20),
                "bilinear_penalty": 5.0,
                "T600_force_fraction": 0.97,
            },
            {
                "projection_mass": 0.55,
                "seed_weights": (1.0 / 3.0,) * 3,
                "bilinear_penalty": 5.0,
                "T600_force_fraction": 0.97,
            },
            {
                "projection_mass": 0.55,
                "seed_weights": default_weights,
                "bilinear_penalty": 5.0,
                "T600_force_fraction": 0.95,
            },
        )
    )
    return tuple(records)


SYSTEMS = candidate_systems()


def load_full_training(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data112, receipt112 = r2z.load_full_training(args)
    fixed2, base_design2, receipt_base2 = r2ad._verified_seed2_base(
        args.seed2_base_root
    )
    full2, receipt_full2 = r2ad._verified_seed2_full(args.seed2_full_root)
    reference2, modes2, coordinates2, base2 = r2ad._seed2_labels(
        args.seed2_labels
    )
    design2 = np.concatenate((base_design2, full2), axis=-1)
    design = np.concatenate(
        (data112["design"][:40], design2, data112["design"][40:]), axis=0
    )
    fixed = np.concatenate(
        (data112["fixed"][:40], fixed2, data112["fixed"][40:]), axis=0
    )
    reference = np.concatenate(
        (data112["reference"][:40], reference2, data112["reference"][40:]),
        axis=0,
    )
    modes = np.concatenate((data112["modes"], modes2), axis=0)
    coordinates = np.concatenate((data112["coordinates"], coordinates2), axis=0)
    base = np.concatenate((data112["base"], base2), axis=0)
    if design.shape != (132, 72, 3, 577):
        raise ValueError("R2AF full seed012 design shape changed")
    return {
        "design": design,
        "fixed": fixed,
        "reference": reference,
        "modes": modes,
        "coordinates": coordinates,
        "base": base,
    }, {
        "seed01_thermal": receipt112,
        "seed2_base": receipt_base2,
        "seed2_full": receipt_full2,
        "seed2_labels_sha256": file_sha256(args.seed2_labels),
    }


def coefficient_matrix(
    system: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    normalizer, gram, rhs = system
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = np.finfo(float).eps * len(normalizer) * max(
        float(eigenvalues[-1]), 1.0
    )
    if eigenvalues[0] < -10.0 * tolerance:
        raise ValueError("R2AF Gram is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_rhs = eigenvectors.T @ rhs
    normalized = eigenvectors @ (
        projected_rhs[:, None]
        / (eigenvalues[:, None] + np.asarray(ALPHAS)[None, :])
    )
    return r2ad.FORCE_SCALE * normalized / normalizer[:, None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=r2z.DEFAULT_AGGREGATE)
    parser.add_argument("--failed-root", type=Path, default=r2z.DEFAULT_R2Y_FAILED)
    parser.add_argument(
        "--seed1-selected-root", type=Path, default=r2z.DEFAULT_SEED1_SELECTED
    )
    parser.add_argument("--thermal-full-root", type=Path, default=r2z.DEFAULT_THERMAL_FULL)
    parser.add_argument("--seed1-full-root", type=Path, default=r2z.DEFAULT_SEED1_FULL)
    parser.add_argument("--seed2-labels", type=Path, default=r2ad.DEFAULT_SEED2_LABELS)
    parser.add_argument("--seed2-base-root", type=Path, default=r2ad.DEFAULT_SEED2_BASE)
    parser.add_argument("--seed2-full-root", type=Path, default=r2ad.DEFAULT_SEED2_FULL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    data, input_receipt = load_full_training(args)
    audit = design_audit(data["design"])
    if not audit["pass"]:
        raise ValueError(f"R2AF full design audit failed: {audit}")
    statistics = r2ad.fold_raw_statistics(data)
    records = []
    best_score = np.inf
    best_prediction: np.ndarray | None = None
    best_record: dict[str, Any] | None = None
    for system_id, specification in enumerate(SYSTEMS):
        oof = np.full((132, 72, 3, len(ALPHAS)), np.nan)
        for hold_id, fold_tuple in enumerate(r2ad.FOLDS):
            hold = np.asarray(fold_tuple, dtype=int)
            coefficient = coefficient_matrix(
                r2ad.normal_equations(
                    statistics,
                    tuple(value for value in range(4) if value != hold_id),
                    float(specification["projection_mass"]),
                    specification["seed_weights"],
                    float(specification["bilinear_penalty"]),
                    float(specification["T600_force_fraction"]),
                )
            )
            oof[hold] = data["fixed"][hold][..., None] + np.einsum(
                "natk,kq->natq", data["design"][hold], coefficient, optimize=True
            )
        scores = r2ad.vectorized_scores(oof, data)
        for alpha_id, score in enumerate(scores):
            record = {
                "raw_gate_score": float(score),
                "system_id": system_id,
                "projection_mass": specification["projection_mass"],
                "seed_projection_weights": {
                    name: float(weight)
                    for name, weight in zip(
                        r2ad.SEED_RANGES,
                        specification["seed_weights"],
                        strict=True,
                    )
                },
                "bilinear_penalty": specification["bilinear_penalty"],
                "T600_force_fraction": specification["T600_force_fraction"],
                "alpha": ALPHAS[alpha_id],
            }
            records.append(record)
            if score < best_score:
                best_score = float(score)
                best_prediction = oof[..., alpha_id].copy()
                best_record = record
        print(
            json.dumps(
                {
                    "system_complete": system_id + 1,
                    "system_count": len(SYSTEMS),
                    "best_score_so_far": best_score,
                }
            ),
            flush=True,
        )
    if best_prediction is None or best_record is None:
        raise RuntimeError("R2AF full-space scan produced no candidate")
    metrics = r2ad.scalar_metrics(best_prediction, data)
    if abs(metrics["raw_gate_score"] - best_score) > 1.0e-10:
        raise ValueError("R2AF vectorized/scalar score mismatch")
    records.sort(key=lambda item: (item["raw_gate_score"], item["alpha"]))
    status = (
        "R2AF_FULL512_DEVELOPMENT_GATE_PASSED"
        if metrics["passes_fixed_gate"]
        else "R2AF_FULL512_DEVELOPMENT_GATE_FAILED"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "best_OOF_predicted_force_eV_A.npy"
    np.save(prediction_path, np.asarray(best_prediction, dtype="<f8"), allow_pickle=False)
    summary = {
        "format": "graphene_r2af_seed012_full512_development_v1",
        "status": status,
        "scope": "complete frozen 512-column bilinear capacity diagnostic; seed012 development training; 525 K unopened",
        "design_audit": audit,
        "system_count": len(SYSTEMS),
        "candidate_count": len(SYSTEMS) * len(ALPHAS),
        "best": {**best_record, "full_gate": metrics},
        "top_candidates": records[:200],
        "input_receipt": input_receipt,
        "prediction_sha256": file_sha256(prediction_path),
        "prediction_raw_sha256": r2r1.raw_array_sha256(best_prediction, "<f8"),
        "unseen_525K_access": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (output / "DONE").write_text(status + "\n")
    print(json.dumps({"status": status, "best": summary["best"]}, indent=2))


if __name__ == "__main__":
    main()
