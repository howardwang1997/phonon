#!/usr/bin/env python3
"""Test whether frozen conservative short-range corrections are composable.

This is a development-set feasibility diagnostic, not a fitted release model.
It scans global coefficients in

    depth3 + alpha * (lastblock - depth3)
           + beta  * compact_short_bond_expert
           + gamma * high_radial_mace_expert

All terms are scalar-energy models, so every scanned combination remains
conservative.  If no global mixture satisfies the fixed support, thermal, and
harmonic gates, a configuration-dependent conservative router (or a different
representation) is required instead of another amplitude sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from audit_graphene_current_s0_fixed_smearing import (  # noqa: E402
    folded_k_aprime_mode,
    force_metrics,
    load_operator,
)
from evaluate_graphene_r2c_candidates import target_arrays  # noqa: E402
from evaluate_graphene_r2d_additive_adapter import (  # noqa: E402
    evaluate_combination,
    model_predictions,
)
from evaluate_graphene_r2g_short_bond_expert import (  # noqa: E402
    expert_predictions,
)


FIXED_DEGAUSS_RY = 0.0019000869380739254
BOUNDS = {
    "alpha_lastblock": (0.0, 1.4),
    "beta_compact": (-0.5, 2.0),
    "gamma_radial": (-0.5, 1.5),
}


def labelled_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("thermal paths use LABEL=/path")
    label, path = specification.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("thermal paths use LABEL=/path")
    return label, Path(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def values_between(lower: float, upper: float, step: float) -> list[float]:
    count = int(round((upper - lower) / step))
    return [round(lower + index * step, 10) for index in range(count + 1)]


def clipped_values(center: float, radius: float, step: float, bounds) -> list[float]:
    lower = max(bounds[0], center - radius)
    upper = min(bounds[1], center + radius)
    first = np.ceil((lower - bounds[0]) / step - 1.0e-10) * step + bounds[0]
    values = []
    value = first
    while value <= upper + 1.0e-10:
        values.append(round(float(value), 10))
        value += step
    return values


def linear_combination(
    depth: dict[str, tuple[np.ndarray, np.ndarray]],
    lastblock: dict[str, tuple[np.ndarray, np.ndarray]],
    compact: dict[str, tuple[np.ndarray, np.ndarray]],
    radial: dict[str, tuple[np.ndarray, np.ndarray]],
    alpha: float,
    beta: float,
    gamma: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        label: (
            depth[label][0]
            + alpha * (lastblock[label][0] - depth[label][0])
            + beta * compact[label][0]
            + gamma * radial[label][0],
            depth[label][1]
            + alpha * (lastblock[label][1] - depth[label][1])
            + beta * compact[label][1]
            + gamma * radial[label][1],
        )
        for label in depth
    }


def concise_row(alpha: float, beta: float, gamma: float, result: dict) -> dict:
    support = result["support9_development_fit"]
    thermal = result["thermal_development"]
    return {
        "coefficients": {
            "alpha_lastblock": alpha,
            "beta_compact": beta,
            "gamma_radial": gamma,
        },
        "passes_all_development_gates": result["passes_all_development_gates"],
        "normalized_worst_gate_score": result["normalized_worst_gate_score"],
        "support_force_RMSE_meV_A": support["force_error"]["RMSE_meV_A"],
        "support_force_max_abs_meV_A": support["force_error"]["max_abs_meV_A"],
        "support_Aprime_RMS_meV_A": support["Aprime_projection"]["force_error"][
            "RMS_meV_A"
        ],
        "support_Aprime_slope_relative_error_abs": abs(
            support["Aprime_projection"]["relative_slope_error"]
        ),
        "support_centered_energy_RMSE_meV_config": support[
            "delta_energy_error_after_global_offset"
        ]["RMSE_meV_config"],
        "support_ESS_fraction": support["importance_weight_ESS_fraction"],
        "harmonic_force_RMSE_meV_A": result["harmonic_replay"][
            "total_force_error"
        ]["RMSE_meV_A"],
        "thermal_force_RMSE_meV_A": {
            label: thermal[label]["total_force_error"]["RMSE_meV_A"]
            for label in ("T300", "T450", "T600")
        },
        "thermal_force_max_abs_meV_A": {
            label: thermal[label]["total_force_error"]["max_abs_meV_A"]
            for label in ("T300", "T450", "T600")
        },
        "normalized_gate_ratios": result["normalized_gate_ratios"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--depth3-model", type=Path, required=True)
    parser.add_argument("--lastblock-model", type=Path, required=True)
    parser.add_argument("--compact-expert", type=Path, required=True)
    parser.add_argument("--radial-adapter", type=Path, required=True)
    parser.add_argument("--support-data", type=Path, required=True)
    parser.add_argument("--thermal", action="append", type=labelled_path, required=True)
    parser.add_argument("--harmonic", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--corrected-result", type=Path, required=True)
    parser.add_argument("--coarse-step", type=float, default=0.1)
    parser.add_argument("--refine-step", type=float, default=0.02)
    parser.add_argument("--refine-seeds", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.coarse_step <= 0.0 or args.refine_step <= 0.0:
        raise ValueError("scan steps must be positive")
    if args.refine_step >= args.coarse_step:
        raise ValueError("refine-step must be smaller than coarse-step")
    thermal_paths = dict(args.thermal)
    if len(thermal_paths) != len(args.thermal):
        raise ValueError("thermal labels must be unique")
    if set(thermal_paths) != {"T300", "T450", "T600"}:
        raise ValueError("thermal inputs must be labelled T300, T450, and T600")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    datasets = {
        "support": read(args.support_data, index=":"),
        **{label: read(path, index=":") for label, path in thermal_paths.items()},
        "harmonic": read(args.harmonic, index=":"),
    }
    if len(datasets["support"]) != 9:
        raise ValueError("support development set must contain nine structures")

    _, reference, cell, operator_degauss = load_operator(args.operator)
    if abs(operator_degauss - FIXED_DEGAUSS_RY) > 5.0e-10:
        raise ValueError("mixture scan requires the frozen T300 q6 operator")
    mode, mode_provenance = folded_k_aprime_mode(
        args.background, args.corrected_result, reference, cell
    )

    print("predicting frozen base", flush=True)
    base = model_predictions(args.base_model, args.device, datasets)
    print("predicting depth-3", flush=True)
    depth = model_predictions(args.depth3_model, args.device, datasets)
    print("predicting last-block candidate", flush=True)
    lastblock = model_predictions(args.lastblock_model, args.device, datasets)
    print("predicting compact short-bond expert", flush=True)
    compact, compact_metadata = expert_predictions(
        args.compact_expert, device, datasets
    )
    print("predicting high-radial MACE expert", flush=True)
    radial = model_predictions(args.radial_adapter, args.device, datasets)

    base_harmonic_errors = []
    for index, structure in enumerate(datasets["harmonic"]):
        target_total, long_force = target_arrays(
            structure, base["harmonic"][1][index]
        )
        base_harmonic_errors.append(
            base["harmonic"][1][index] + long_force - target_total
        )
    base_harmonic = force_metrics(np.asarray(base_harmonic_errors))
    harmonic_limit = 1.25 * base_harmonic["RMSE_meV_A"]
    zero = {
        label: (np.zeros_like(values[0]), np.zeros_like(values[1]))
        for label, values in depth.items()
    }

    rows: dict[tuple[float, float, float], dict] = {}
    full_best = None

    def evaluate(alpha: float, beta: float, gamma: float) -> None:
        nonlocal full_best
        key = (round(alpha, 10), round(beta, 10), round(gamma, 10))
        if key in rows:
            return
        candidate = linear_combination(
            depth, lastblock, compact, radial, alpha, beta, gamma
        )
        result = evaluate_combination(
            datasets,
            base,
            candidate,
            zero,
            mode,
            reference,
            cell,
            harmonic_limit,
        )
        row = concise_row(alpha, beta, gamma, result)
        rows[key] = row
        if full_best is None or row["normalized_worst_gate_score"] < full_best[0]:
            full_best = (row["normalized_worst_gate_score"], key, result)

    landmarks = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
    ]
    for point in landmarks:
        evaluate(*point)

    axes = {
        name: values_between(bounds[0], bounds[1], args.coarse_step)
        for name, bounds in BOUNDS.items()
    }
    total_coarse = int(np.prod([len(values) for values in axes.values()]))
    print(f"coarse scan: {total_coarse} points", flush=True)
    for index, point in enumerate(
        itertools.product(
            axes["alpha_lastblock"],
            axes["beta_compact"],
            axes["gamma_radial"],
        ),
        start=1,
    ):
        evaluate(*point)
        if index % 1000 == 0:
            print(f"coarse progress {index}/{total_coarse}", flush=True)

    coarse_ranking = sorted(
        rows.values(), key=lambda row: row["normalized_worst_gate_score"]
    )
    seeds = coarse_ranking[: args.refine_seeds]
    before_refine = len(rows)
    for seed in seeds:
        coefficients = seed["coefficients"]
        local_axes = {
            name: clipped_values(
                coefficients[name], args.coarse_step, args.refine_step, BOUNDS[name]
            )
            for name in BOUNDS
        }
        for point in itertools.product(
            local_axes["alpha_lastblock"],
            local_axes["beta_compact"],
            local_axes["gamma_radial"],
        ):
            evaluate(*point)
    print(f"refined with {len(rows) - before_refine} new points", flush=True)

    ranking = sorted(
        rows.values(), key=lambda row: row["normalized_worst_gate_score"]
    )
    feasible = [row for row in ranking if row["passes_all_development_gates"]]
    gate_names = list(ranking[0]["normalized_gate_ratios"])
    minima_by_gate = {
        gate: min(
            rows.values(), key=lambda row: row["normalized_gate_ratios"][gate]
        )
        for gate in gate_names
    }
    landmark_rows = {
        f"alpha={alpha:g},beta={beta:g},gamma={gamma:g}": rows[
            (alpha, beta, gamma)
        ]
        for alpha, beta, gamma in landmarks
    }
    assert full_best is not None
    best_key = full_best[1]
    summary = {
        "status": "R2H_frozen_conservative_mixture_feasibility_complete",
        "interpretation_scope": (
            "development-set global-coefficient feasibility only; this scan does "
            "not train or validate a transferable configuration-dependent router"
        ),
        "new_DFT_labels": 0,
        "long_range_model_modified": False,
        "combination": (
            "depth3 + alpha*(lastblock-depth3) + beta*compact + gamma*radial"
        ),
        "fixed_condition": {
            "lattice_temperature_K": 450.0,
            "smearing": "fermi-dirac",
            "degauss_Ry": operator_degauss,
        },
        "scan": {
            "bounds": BOUNDS,
            "coarse_step": args.coarse_step,
            "refine_step": args.refine_step,
            "refine_seeds": args.refine_seeds,
            "n_unique_points": len(rows),
            "n_feasible_points": len(feasible),
        },
        "best_coefficients": ranking[0]["coefficients"],
        "best_passes_all_development_gates": ranking[0][
            "passes_all_development_gates"
        ],
        "best": ranking[0],
        "best_full_metrics": full_best[2],
        "top_candidates": ranking[: args.top_k],
        "first_feasible_candidates": feasible[: args.top_k],
        "landmarks": landmark_rows,
        "minimum_each_gate": minima_by_gate,
        "frozen_base_harmonic_force": base_harmonic,
        "harmonic_limit_meV_A": harmonic_limit,
        "Aprime_mode": mode_provenance,
        "models": {
            "base": {"path": str(args.base_model), "sha256": sha256(args.base_model)},
            "depth3": {
                "path": str(args.depth3_model),
                "sha256": sha256(args.depth3_model),
            },
            "lastblock": {
                "path": str(args.lastblock_model),
                "sha256": sha256(args.lastblock_model),
            },
            "compact": {
                "path": str(args.compact_expert),
                "sha256": sha256(args.compact_expert),
                "metadata": compact_metadata,
            },
            "radial": {
                "path": str(args.radial_adapter),
                "sha256": sha256(args.radial_adapter),
            },
        },
        "inputs": {
            "support": {"path": str(args.support_data), "sha256": sha256(args.support_data)},
            "thermal": {
                label: {"path": str(path), "sha256": sha256(path)}
                for label, path in thermal_paths.items()
            },
            "harmonic": {"path": str(args.harmonic), "sha256": sha256(args.harmonic)},
            "operator": {"path": str(args.operator), "sha256": sha256(args.operator)},
            "background": {"path": str(args.background), "sha256": sha256(args.background)},
            "corrected_result": {
                "path": str(args.corrected_result),
                "sha256": sha256(args.corrected_result),
            },
        },
    }
    if ranking[0]["coefficients"] != {
        "alpha_lastblock": best_key[0],
        "beta_compact": best_key[1],
        "gamma_radial": best_key[2],
    }:
        raise AssertionError("best-row bookkeeping mismatch")
    atomic_json(args.output, summary)
    print(
        json.dumps(
            {
                "n_unique_points": len(rows),
                "n_feasible_points": len(feasible),
                "best": ranking[0],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
