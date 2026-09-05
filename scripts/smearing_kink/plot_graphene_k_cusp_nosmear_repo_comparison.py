#!/usr/bin/env python3
"""Plot the local graphene A1' cusp for direct DFPT and the repository method.

The comparison is deliberately restricted to the predeclared K window.  It
shows the static short-range MLIP background, the same background after the
frozen q-space long-range correction, and direct optimized-tetrahedron DFPT
points.  Blind-holdout points can be added without changing the frozen curve.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_graphene_k_cusp_nosmear_method as method


ROOT = Path(__file__).resolve().parents[2]


def q_reduced(direction: str, distance: float) -> np.ndarray:
    if direction == "K":
        h, k = 1.0 / 3.0, 1.0 / 3.0
    elif direction == "KG":
        h = k = (1.0 - distance) / 3.0
    elif direction == "KM":
        h, k = (1.0 + distance) / 3.0, (1.0 - 2.0 * distance) / 3.0
    else:
        raise ValueError(f"unknown direction: {direction}")
    return np.asarray([h, k, 0.0])


def signed_distance(direction: str, distance: float) -> float:
    if direction == "KM":
        return -distance
    if direction in {"K", "KG"}:
        return distance
    raise ValueError(f"unknown direction: {direction}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_frozen_model(path: Path) -> tuple[dict, str, np.ndarray]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_holdout":
        raise ValueError("development model has not passed its freeze gate")
    model_path = Path(manifest["model_artifact"]["path"])
    if not model_path.is_absolute():
        model_path = path.parent / model_path.name
    if method.digest(model_path) != manifest["model_artifact"]["sha256"]:
        raise ValueError("frozen model hash mismatch")
    with np.load(model_path, allow_pickle=False) as payload:
        model_name = str(payload["model_name"].item())
        coefficients = np.asarray(payload["coefficients_cm2"], float)
    return manifest, model_name, coefficients


def dense_curve(model_name: str, coefficients: np.ndarray, npoints: int) -> dict:
    distance = np.linspace(0.0, 0.023, npoints)
    branches = {}
    hermiticity = 0.0
    replay = 0.0
    source_metadata = None
    for direction in ("KM", "KG"):
        rows = [
            {
                "direction": direction,
                "delta": float(value),
                "q_reduced": q_reduced(direction, float(value)),
            }
            for value in distance
        ]
        prediction = method.predict_frequency(model_name, coefficients, rows)
        projector = method.static_projector_replay(rows, prediction)
        metadata = {
            "static_short_fc2": projector["static_short_fc2"],
            "static_short_fc2_sha256": projector["static_short_fc2_sha256"],
            "operator_geometry": projector["operator_geometry"],
            "operator_geometry_sha256": projector["operator_geometry_sha256"],
        }
        if source_metadata is None:
            source_metadata = metadata
        elif metadata != source_metadata:
            raise ValueError("repository baseline provenance changed between directions")
        hermiticity = max(hermiticity, float(projector["Hermiticity_max_abs"]))
        replay = max(
            replay, float(projector["rank_one_replay_max_abs_cm-1"])
        )
        branches[direction] = {
            "distance": distance,
            "short_mlip_cm-1": np.asarray(projector["baseline_cm-1"], float),
            "repo_method_cm-1": np.asarray(projector["corrected_cm-1"], float),
        }
    km = branches["KM"]
    kg = branches["KG"]
    return {
        "x": np.concatenate((-km["distance"][:0:-1], kg["distance"])),
        "short_mlip_cm-1": np.concatenate(
            (km["short_mlip_cm-1"][:0:-1], kg["short_mlip_cm-1"])
        ),
        "repo_method_cm-1": np.concatenate(
            (km["repo_method_cm-1"][:0:-1], kg["repo_method_cm-1"])
        ),
        "Hermiticity_max_abs": hermiticity,
        "rank_one_replay_max_abs_cm-1": replay,
        "source_metadata": source_metadata,
    }


def direct_points(
    development_path: Path, holdout_path: Path | None
) -> tuple[list[dict], float]:
    development = []
    k_frequency = None
    for row in read_csv(development_path):
        direction = row["direction"]
        distance = float(row["delta"])
        frequency = float(row["frequency_cm-1"])
        if direction == "K":
            k_frequency = frequency
        development.append(
            {
                "set": "development",
                "direction": direction,
                "distance": distance,
                "signed_distance": signed_distance(direction, distance),
                "direct_DFPT_cm-1": frequency,
            }
        )
    if k_frequency is None:
        raise ValueError("development points do not contain the shared K anchor")

    holdout = []
    if holdout_path is not None:
        for row in read_csv(holdout_path):
            direction = row["direction"]
            distance = float(row["delta"])
            recorded_k = float(row["K_frequency_cm-1"])
            if not np.isclose(recorded_k, k_frequency, atol=1.0e-8):
                raise ValueError("holdout and development use different K anchors")
            holdout.append(
                {
                    "set": "blind_holdout",
                    "direction": direction,
                    "distance": distance,
                    "signed_distance": signed_distance(direction, distance),
                    "direct_DFPT_cm-1": float(row["frequency_cm-1"]),
                }
            )
    return development + holdout, float(k_frequency)


def write_curve_csv(path: Path, curve: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "signed_equal_distance_from_K",
                "short_range_MLIP_cm-1",
                "MLIP_plus_qspace_long_range_cm-1",
            ],
        )
        writer.writeheader()
        for index, value in enumerate(curve["x"]):
            writer.writerow(
                {
                    "signed_equal_distance_from_K": float(value),
                    "short_range_MLIP_cm-1": float(curve["short_mlip_cm-1"][index]),
                    "MLIP_plus_qspace_long_range_cm-1": float(
                        curve["repo_method_cm-1"][index]
                    ),
                }
            )


def make_figure(
    curve: dict,
    points: list[dict],
    k_frequency: float,
    model_name: str,
    output: Path,
) -> None:
    x = np.asarray(curve["x"], float)
    short = np.asarray(curve["short_mlip_cm-1"], float)
    repo = np.asarray(curve["repo_method_cm-1"], float)
    center = int(np.argmin(np.abs(x)))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    fig, axes = plt.subplots(2, 1, figsize=(7.3, 6.2), sharex=True)
    for axis, centered in zip(axes, (False, True), strict=True):
        short_y = short - short[center] if centered else short
        repo_y = repo - repo[center] if centered else repo
        axis.plot(
            x,
            short_y,
            "--",
            color="#888888",
            lw=1.4,
            label="short-range MLIP",
        )
        axis.plot(
            x,
            repo_y,
            "-",
            color="#1665A8",
            lw=1.8,
            label="MLIP + q-space\nlong-range correction",
        )
        for point_set, marker, face, label in (
            ("development", "o", "#111111", "direct DFPT development"),
            ("blind_holdout", "D", "white", "direct DFPT blind holdout"),
        ):
            selected = [row for row in points if row["set"] == point_set]
            if not selected:
                continue
            y = np.asarray([row["direct_DFPT_cm-1"] for row in selected])
            if centered:
                y = y - k_frequency
            axis.scatter(
                [row["signed_distance"] for row in selected],
                y,
                marker=marker,
                s=31,
                facecolors=face,
                edgecolors="#111111",
                linewidths=0.9,
                zorder=4,
                label=label,
            )
        axis.axvline(0.0, color="#C8C8C8", lw=0.8, zorder=0)
        axis.grid(axis="y", color="#E8E8E8", lw=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$A_1'$ frequency (cm$^{-1}$)")
    axes[0].set_title("Graphene K cusp: direct DFPT vs repository method")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    axes[1].set_ylabel(r"relative to K (cm$^{-1}$)")
    axes[1].set_xlabel("equal-distance coordinate around K")
    axes[1].set_xticks([-0.023, 0.0, 0.023], ["K→M", "K", "K→Γ"])
    axes[1].text(
        0.5,
        -0.28,
        "electronic integration: tetrahedra_opt (no degauss)"
        f"; frozen model: {model_name}",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=8.3,
    )
    fig.subplots_adjust(left=0.12, right=0.69, top=0.92, bottom=0.16, hspace=0.16)
    fig.savefig(output, dpi=240, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--development-points", type=Path)
    parser.add_argument("--holdout-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--npoints", type=int, default=161)
    args = parser.parse_args()
    if args.npoints < 3:
        raise ValueError("npoints must be at least 3")
    development_path = args.development_points or (
        args.freeze_manifest.parent / "development_points.csv"
    )
    manifest, model_name, coefficients = load_frozen_model(args.freeze_manifest)
    points, k_frequency = direct_points(
        development_path, args.holdout_predictions
    )
    curve = dense_curve(model_name, coefficients, args.npoints)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "repo_method_blind_comparison" if args.holdout_predictions else "repo_method_development_comparison"
    figure_path = args.output_dir / f"{stem}.png"
    curve_path = args.output_dir / f"{stem}_curve.csv"
    make_figure(curve, points, k_frequency, model_name, figure_path)
    write_curve_csv(curve_path, curve)
    summary = {
        "status": "plotted_blind_holdout" if args.holdout_predictions else "plotted_development_only",
        "scope": "local graphene A1prime K window only",
        "electronic_integration": "tetrahedra_opt (no degauss)",
        "frozen_model": model_name,
        "n_development_points": sum(row["set"] == "development" for row in points),
        "n_blind_holdout_points": sum(row["set"] == "blind_holdout" for row in points),
        "matrix_checks": {
            "Hermiticity_max_abs": curve["Hermiticity_max_abs"],
            "rank_one_replay_max_abs_cm-1": curve[
                "rank_one_replay_max_abs_cm-1"
            ],
        },
        "sources": {
            "freeze_manifest": {
                "path": str(args.freeze_manifest),
                "sha256": method.digest(args.freeze_manifest),
            },
            "development_points": {
                "path": str(development_path),
                "sha256": method.digest(development_path),
            },
            "holdout_predictions": (
                {
                    "path": str(args.holdout_predictions),
                    "sha256": method.digest(args.holdout_predictions),
                }
                if args.holdout_predictions
                else None
            ),
            "short_range_MLIP": {
                "path": curve["source_metadata"]["static_short_fc2"],
                "sha256": curve["source_metadata"]["static_short_fc2_sha256"],
            },
            "qspace_operator_geometry": {
                "path": curve["source_metadata"]["operator_geometry"],
                "sha256": curve["source_metadata"]["operator_geometry_sha256"],
            },
        },
        "outputs": [str(figure_path), str(figure_path.with_suffix(".pdf")), str(curve_path)],
    }
    method.atomic_json(args.output_dir / f"{stem}_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
