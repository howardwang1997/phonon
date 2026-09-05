#!/usr/bin/env python3
"""Development-only experiment for a reciprocal-space Kohn correction.

This script uses only the pre-declared direct-DFPT development set
(``degauss={0.01,0.02,0.04,0.06,0.08} Ry``).  It models the absolute direct-DFPT
squared frequency with a thermally rounded cusp, then realizes the difference
from the frozen baseline as a dynamical-matrix projector.  A smooth polynomial
and a cusp-plus-polynomial form remain controls.  Selection uses grouped
q-point cross-validation plus leave-one-smearing-out interpolation tests.

It does *not* read or evaluate the final holdout defined in
``docs/GRAPHENE_KOHN_QSPACE_FIX_2026-07-23.md``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_graphene_dfpt_linecuts as dfpt  # noqa: E402
from compare_graphene_dfpt_method import load_cached_fc  # noqa: E402
import friedel_module as fm  # noqa: E402
import graphene_p0_deploy as p0  # noqa: E402
from qe_dyn import load_qe_dyn  # noqa: E402


INDIR = ROOT / "results" / "p0_graphene_dfpt"
OUTDIR = ROOT / "results" / "p0_graphene_qspace"
DEVELOPMENT_DFPT = OUTDIR / "development_dfpt"
ORIGINAL_DGS = (0.01, 0.04, 0.08)
SUPPLEMENTAL_CAMPAIGNS = {0.02: "DEV_A", 0.06: "DEV_B"}
DEVELOPMENT_DGS = (0.01, 0.02, 0.04, 0.06, 0.08)
WIDTH_SCALES = (0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
CANDIDATES = ("poly2", "rounded_cusp", "rounded_cusp_poly2")
TEMPERATURE_POLYNOMIAL_DEGREE = 3


def load_development_rows() -> list[dict[str, float | str]]:
    rows = []
    with (INDIR / "graphene_dfpt_method_linecuts.csv").open() as handle:
        for raw in csv.DictReader(handle):
            if raw["law"] != "log_interp":
                continue
            dg = float(raw["degauss_Ry"])
            if dg not in ORIGINAL_DGS:
                continue
            rows.append(
                {
                    "degauss_Ry": dg,
                    "region": raw["region"],
                    "t_GK": float(raw["t_GK"]),
                    "dft_cm": float(raw["DFPT_f6_cm-1"]),
                    "base_cm": float(raw["method_f6_cm-1"]),
                    "direct_source": "archived_primary_DFPT",
                }
            )

    for expected_dg, campaign in SUPPLEMENTAL_CAMPAIGNS.items():
        path = DEVELOPMENT_DFPT / campaign / f"dfpt_{campaign}.csv"
        if not path.is_file():
            raise RuntimeError(
                f"missing supplemental development data: {path}; "
                "the final holdout remains locked until both DEV campaigns finish"
            )
        with path.open() as handle:
            for raw in csv.DictReader(handle):
                dg = float(raw["degauss_Ry"])
                if (
                    raw["campaign"] != campaign
                    or not np.isclose(dg, expected_dg)
                    or int(raw["kgrid"]) != 32
                ):
                    raise RuntimeError(f"unexpected row in {path}: {raw}")
                rows.append(
                    {
                        "degauss_Ry": dg,
                        "region": raw["region"],
                        "t_GK": float(raw["t_GK"]),
                        "dft_cm": float(raw["f6_cm"]),
                        "base_cm": float("nan"),
                        "direct_source": campaign,
                    }
                )

    expected = len(DEVELOPMENT_DGS) * (5 + 9)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} development rows, found {len(rows)}")
    keys = [
        (float(row["degauss_Ry"]), str(row["region"]), float(row["t_GK"]))
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate development (degauss, region, t) rows")

    # Recompute every baseline from the frozen FC caches.  For the three
    # archived line cuts, require exact agreement with their stored values;
    # this makes the two supplemental temperatures use the identical path.
    for dg in DEVELOPMENT_DGS:
        for region, expected_count in (("G", 5), ("K", 9)):
            data = grouped(rows, region, dg)
            if len(data) != expected_count:
                raise RuntimeError(
                    f"expected {expected_count} rows for {region}, dg={dg}; "
                    f"found {len(data)}"
                )
            t = np.array([float(row["t_GK"]) for row in data])
            frequencies, _ = method_dynamical_matrices(region, dg, t)
            for row, value in zip(data, frequencies[:, -1]):
                archived = float(row["base_cm"])
                if np.isfinite(archived) and abs(archived - float(value)) > 1e-6:
                    raise RuntimeError(
                        f"frozen baseline replay mismatch for {region}, dg={dg}, "
                        f"t={row['t_GK']}: {archived} vs {value}"
                    )
                row["base_cm"] = float(value)
    return rows


def distance_from_anomaly(region: str, t: np.ndarray) -> np.ndarray:
    return np.asarray(t, float) if region == "G" else np.abs(np.asarray(t, float) - 1.0)


def features(kind: str, x: np.ndarray, dg: float, width_scale: float | None) -> np.ndarray:
    x = np.asarray(x, float)
    columns = [np.ones_like(x)]
    if kind in ("rounded_cusp", "rounded_cusp_poly2"):
        if width_scale is None:
            raise ValueError(f"{kind} needs a width scale")
        epsilon = max(float(width_scale) * float(dg), 1e-10)
        # Finite-T regularisation of |q-q0|: quadratic at the centre and
        # asymptotically linear outside the thermal width.
        rounded_abs = np.sqrt(x * x + epsilon * epsilon) - epsilon
        columns.append(rounded_abs)
    if kind in ("poly2", "rounded_cusp_poly2"):
        columns.append(x * x)
    return np.column_stack(columns)


def fit_coefficients(
    rows: list[dict[str, float | str]],
    kind: str,
    width_scale: float | None,
) -> np.ndarray:
    region = str(rows[0]["region"])
    dg = float(rows[0]["degauss_Ry"])
    t = np.array([float(row["t_GK"]) for row in rows])
    x = distance_from_anomaly(region, t)
    target_lambda = np.array([float(row["dft_cm"]) ** 2 for row in rows])
    design = features(kind, x, dg, width_scale)
    coeff, *_ = np.linalg.lstsq(design, target_lambda, rcond=None)
    return coeff


def corrected_frequency(
    rows: list[dict[str, float | str]],
    kind: str,
    width_scale: float | None,
    coeff: np.ndarray,
) -> np.ndarray:
    region = str(rows[0]["region"])
    dg = float(rows[0]["degauss_Ry"])
    t = np.array([float(row["t_GK"]) for row in rows])
    x = distance_from_anomaly(region, t)
    target_lambda = features(kind, x, dg, width_scale) @ coeff
    return np.sqrt(np.maximum(target_lambda, 0.0))


def grouped(
    rows: list[dict[str, float | str]], region: str, dg: float
) -> list[dict[str, float | str]]:
    result = [
        row for row in rows
        if str(row["region"]) == region and float(row["degauss_Ry"]) == dg
    ]
    result.sort(key=lambda row: float(row["t_GK"]))
    return result


def qpoint_cross_validation(
    rows: list[dict[str, float | str]],
    region: str,
    kind: str,
    width_scale: float | None,
) -> float:
    errors = []
    for dg in DEVELOPMENT_DGS:
        data = grouped(rows, region, dg)
        x = distance_from_anomaly(
            region, np.array([float(row["t_GK"]) for row in data])
        )
        # Hold out equal-|q-q0| groups together, so K's left/right symmetry pair
        # never leaks from train to validation.
        for x_hold in sorted(set(np.round(x, 12))):
            train = [row for row, xi in zip(data, x) if not np.isclose(xi, x_hold)]
            valid = [row for row, xi in zip(data, x) if np.isclose(xi, x_hold)]
            coeff = fit_coefficients(train, kind, width_scale)
            pred = corrected_frequency(valid, kind, width_scale, coeff)
            target = np.array([float(row["dft_cm"]) for row in valid])
            errors.extend(np.abs(pred - target).tolist())
    return float(np.mean(errors))


def coefficient_law(
    coeff_by_dg: dict[float, np.ndarray], dg: float
) -> np.ndarray:
    """Evaluate the frozen cubic-in-degauss coefficient law.

    With all five development nodes this is a least-squares cubic.  During
    leave-one-smearing-out validation, the four surviving nodes uniquely
    determine the cubic, so the held point is never used.
    """
    anchors = np.array(sorted(coeff_by_dg), float)
    values = np.stack([coeff_by_dg[value] for value in anchors])
    degree = min(TEMPERATURE_POLYNOMIAL_DEGREE, len(anchors) - 1)
    return np.array(
        [
            np.polyval(np.polyfit(anchors, values[:, column], degree), dg)
            for column in range(values.shape[1])
        ]
    )


def smearing_interpolation_cross_validation(
    rows: list[dict[str, float | str]],
    region: str,
    kind: str,
    width_scale: float | None,
) -> tuple[float, dict[float, float]]:
    errors = []
    by_smearing = {}
    # Each interior temperature is removed completely.  Its coefficient vector
    # is then predicted by the same cubic-in-degauss law that will be used for
    # final predictions.
    for held_dg in DEVELOPMENT_DGS[1:-1]:
        training = {
            dg: fit_coefficients(grouped(rows, region, dg), kind, width_scale)
            for dg in DEVELOPMENT_DGS
            if dg != held_dg
        }
        predicted_coeff = coefficient_law(training, held_dg)
        held = grouped(rows, region, held_dg)
        pred = corrected_frequency(held, kind, width_scale, predicted_coeff)
        target = np.array([float(row["dft_cm"]) for row in held])
        held_errors = np.abs(pred - target)
        by_smearing[held_dg] = float(np.mean(held_errors))
        errors.extend(held_errors.tolist())
    return float(np.mean(errors)), by_smearing


def select_candidates(rows: list[dict[str, float | str]]):
    selected, unconstrained, all_scores = {}, {}, {}
    for region in ("G", "K"):
        candidates = []
        for kind in CANDIDATES:
            widths = (None,) if kind == "poly2" else WIDTH_SCALES
            for width in widths:
                q_mae = qpoint_cross_validation(rows, region, kind, width)
                t_mae, t_by_dg = smearing_interpolation_cross_validation(
                    rows, region, kind, width
                )
                nparam = features(kind, np.array([0.0]), 0.01, width).shape[1]
                # Both tests have equal weight.  The small fixed complexity
                # penalty prevents a numerically immaterial third coefficient
                # from winning by construction.
                score = 0.5 * (q_mae + t_mae) + 0.10 * max(0, nparam - 2)
                candidates.append(
                    {
                        "kind": kind,
                        "width_scale_per_Ry": width,
                        "n_parameters_per_smearing": nparam,
                        "q_group_CV_MAE_cm-1": q_mae,
                        "smearing_LOOCV_MAE_cm-1": t_mae,
                        "smearing_LOOCV_by_degauss_cm-1": {
                            f"{dg:g}": value for dg, value in t_by_dg.items()
                        },
                        "selection_score": score,
                    }
                )
        candidates.sort(key=lambda item: item["selection_score"])
        all_scores[region] = candidates
        unconstrained[region] = candidates[0]
        # The scientific objective is a Kohn-cusp model, so the smooth
        # polynomial is a falsification control rather than an eligible primary
        # model.  Select the best two-parameter rounded-cusp candidate using
        # the same score; no final-holdout result is available at this point.
        selected[region] = next(
            item for item in candidates if item["kind"] == "rounded_cusp"
        )
    return selected, unconstrained, all_scores


def method_dynamical_matrices(region: str, dg: float, t: np.ndarray):
    cfg = p0.DATASETS["8x8"]
    dft_paths = p0.discover_yamls(cfg)
    ph = fm.load_ph(dft_paths[p0.require_point(dft_paths, cfg.background_dg)])
    fc = load_cached_fc(
        ROOT / "results" / "p0_graphene_prospective" / "cache", dg
    )
    ph.force_constants = fc
    qpoints = np.array([[value / 3.0, value / 3.0, 0.0] for value in t])
    ph.run_qpoints(qpoints, with_dynamical_matrices=True)
    result = ph.get_qpoints_dict()
    return (
        np.sort(np.asarray(result["frequencies"], float), axis=1) * p0.CM,
        np.asarray(result["dynamical_matrices"], complex),
    )


def apply_top_projector(
    matrix: np.ndarray,
    delta_lambda_cm2: float,
    frequencies_cm: np.ndarray,
    gamma_degenerate: bool,
) -> np.ndarray:
    herm = (matrix + matrix.conj().T) / 2
    eig, vec = np.linalg.eigh(herm)
    scale_mask = (eig > 1e-10) & (frequencies_cm > 1e-6)
    if not np.any(scale_mask):
        raise ValueError("no positive modes available for dynamical-matrix scaling")
    scale = float(np.median(frequencies_cm[scale_mask] ** 2 / eig[scale_mask]))
    indices = (-2, -1) if gamma_degenerate else (-1,)
    correction = np.zeros_like(herm)
    share = delta_lambda_cm2 if not gamma_degenerate else delta_lambda_cm2
    for index in indices:
        vector = vec[:, index]
        correction += (share / scale) * np.outer(vector, vector.conj())
    return herm + correction


def projector_replay(
    rows: list[dict[str, float | str]],
    selected: dict,
    coeffs: dict[str, dict[float, np.ndarray]],
) -> float:
    max_error = 0.0
    for region in ("G", "K"):
        model = selected[region]
        for dg in DEVELOPMENT_DGS:
            data = grouped(rows, region, dg)
            t = np.array([float(row["t_GK"]) for row in data])
            base_freq, matrices = method_dynamical_matrices(region, dg, t)
            archived_base = np.array([float(row["base_cm"]) for row in data])
            if np.max(np.abs(base_freq[:, -1] - archived_base)) > 1e-6:
                raise RuntimeError(f"baseline dynamical matrix replay failed for {region}, {dg}")
            x = distance_from_anomaly(region, t)
            target_lambda = features(
                model["kind"], x, dg, model["width_scale_per_Ry"]
            ) @ coeffs[region][dg]
            delta = target_lambda - archived_base ** 2
            scalar_pred = np.sqrt(np.maximum(target_lambda, 0.0))
            for index, (matrix, dlambda) in enumerate(zip(matrices, delta)):
                corrected = apply_top_projector(
                    matrix,
                    float(dlambda),
                    base_freq[index],
                    # The Gamma anomaly belongs to the two-dimensional E2g
                    # subspace.  Correct both partners along the short ray;
                    # correcting only the upper sorted branch lets its
                    # uncorrected partner become the apparent f6 branch.
                    gamma_degenerate=(region == "G"),
                )
                eig = np.linalg.eigvalsh(corrected)
                base_eig = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2)
                scale_mask = (base_eig > 1e-10) & (base_freq[index] > 1e-6)
                scale = float(np.median(
                    base_freq[index][scale_mask] ** 2 / base_eig[scale_mask]
                ))
                corrected_freq = np.sign(eig) * np.sqrt(np.abs(eig) * scale)
                max_error = max(
                    max_error, abs(float(corrected_freq[-1]) - float(scalar_pred[index]))
                )
    return max_error


def direct_mode_continuity() -> dict[str, float]:
    result = {}
    for region in ("G", "K"):
        overlaps = []
        for dg in DEVELOPMENT_DGS:
            if dg in ORIGINAL_DGS:
                root = INDIR / "dyn" / "B" / f"dg{dg:g}_k32"
            else:
                campaign = SUPPLEMENTAL_CAMPAIGNS[dg]
                root = DEVELOPMENT_DFPT / campaign / f"dg{dg:g}_k32"
            paths = sorted(
                root.glob(f"{region}_t*/gr.dyn"),
                key=lambda path: float(path.parent.name.split("_t", 1)[1].replace("p", ".")),
            )
            expected_count = 5 if region == "G" else 9
            if len(paths) != expected_count:
                raise RuntimeError(
                    f"expected {expected_count} direct matrices under {root}, "
                    f"found {len(paths)}"
                )
            vectors = [load_qe_dyn(path).eigenvectors[-1].reshape(-1) for path in paths]
            # At Gamma the t=0 E2g pair is degenerate and its individual basis
            # vector is arbitrary; continuity starts at the first nonzero q.
            start = 1 if region == "G" else 0
            for left, right in zip(vectors[start:], vectors[start + 1:]):
                overlaps.append(float(abs(np.vdot(left, right)) ** 2))
        result[region] = min(overlaps)
    return result


def metrics_and_predictions(rows, selected, coeffs):
    predictions, metrics = [], []
    for region in ("G", "K"):
        model = selected[region]
        for dg in DEVELOPMENT_DGS:
            data = grouped(rows, region, dg)
            pred = corrected_frequency(
                data,
                model["kind"],
                model["width_scale_per_Ry"],
                coeffs[region][dg],
            )
            target = np.array([float(row["dft_cm"]) for row in data])
            base = np.array([float(row["base_cm"]) for row in data])
            metrics.append(
                {
                    "region": region,
                    "degauss_Ry": dg,
                    "uncorrected_MAE_cm-1": float(np.mean(np.abs(base - target))),
                    "corrected_development_MAE_cm-1": float(np.mean(np.abs(pred - target))),
                    "max_abs_error_cm-1": float(np.max(np.abs(pred - target))),
                }
            )
            for row, value in zip(data, pred):
                predictions.append(
                    {
                        **row,
                        "corrected_cm": float(value),
                        "delta_lambda_cm-2": (
                            float(value) ** 2 - float(row["base_cm"]) ** 2
                        ),
                        "uncorrected_error_cm-1": float(row["base_cm"]) - float(row["dft_cm"]),
                        "corrected_error_cm-1": float(value) - float(row["dft_cm"]),
                    }
                )
    return predictions, metrics


def make_figure(predictions, selected):
    fig, axes = plt.subplots(2, len(DEVELOPMENT_DGS), figsize=(18.0, 6.8))
    for col, dg in enumerate(DEVELOPMENT_DGS):
        for row_index, region in enumerate(("G", "K")):
            ax = axes[row_index, col]
            data = [
                row for row in predictions
                if row["region"] == region and row["degauss_Ry"] == dg
            ]
            data.sort(key=lambda row: row["t_GK"])
            t = [row["t_GK"] for row in data]
            ax.plot(t, [row["dft_cm"] for row in data], "-o", color="#222222",
                    lw=1.5, ms=4, label="direct DFPT")
            ax.plot(t, [row["base_cm"] for row in data], "--", color="#D55E00",
                    lw=1.4, label="frozen 8x8 method")
            ax.plot(t, [row["corrected_cm"] for row in data], "-.", color="#0072B2",
                    lw=1.5, label="q-space development fit")
            if region == "K":
                ax.axvline(1.0, color="#bbbbbb", lw=0.8)
            ax.set_title(f"{region}, dg={dg:g} Ry")
            ax.set_xlabel(r"$t$ along $q=tK$")
            if col == 0:
                ax.set_ylabel(r"highest optical frequency (cm$^{-1}$)")
            dfpt.style_axis(ax)
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(
        "Graphene q-space Kohn correction — development data only "
        f"(G: {selected['G']['kind']}; K: {selected['K']['kind']})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(OUTDIR / f"qspace_development.{suffix}", dpi=200)
    plt.close(fig)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = load_development_rows()
    selected, unconstrained, scores = select_candidates(rows)
    coeffs: dict[str, dict[float, np.ndarray]] = {}
    for region in ("G", "K"):
        model = selected[region]
        coeffs[region] = {
            dg: fit_coefficients(
                grouped(rows, region, dg),
                model["kind"],
                model["width_scale_per_Ry"],
            )
            for dg in DEVELOPMENT_DGS
        }
    predictions, metrics = metrics_and_predictions(rows, selected, coeffs)
    projector_error = projector_replay(rows, selected, coeffs)
    continuity = direct_mode_continuity()

    with (OUTDIR / "qspace_development_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    with (OUTDIR / "qspace_development_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "status": "development_only_not_frozen",
        "target_leakage": (
            "final holdout is not read; only direct DFPT "
            "dg=0.01/0.02/0.04/0.06/0.08 is used"
        ),
        "selection_rule": (
            "primary is the rounded-cusp candidate minimizing "
            "0.5*(q-group CV MAE + leave-one-smearing-out CV MAE) "
            "+ 0.10 cm^-1 per parameter beyond two; smooth polynomial is "
            "reported as a non-cusp control"
        ),
        "smearing_cross_validation_holdouts_Ry": list(DEVELOPMENT_DGS[1:-1]),
        "target_definition": (
            "absolute direct-DFPT highest-optical squared frequency; the "
            "runtime dynamical-matrix correction is target minus frozen baseline"
        ),
        "selected": selected,
        "unconstrained_numerical_winner_including_smooth_control": unconstrained,
        "candidate_scores": scores,
        "coefficient_interpolation": (
            "least-squares degree-3 polynomial in degauss_Ry; four-node cubic "
            "during leave-one-smearing-out validation"
        ),
        "coefficients_by_degauss": {
            region: {
                f"{dg:g}": coeffs[region][dg].tolist() for dg in DEVELOPMENT_DGS
            }
            for region in ("G", "K")
        },
        "development_metrics": metrics,
        "matrix_projector_replay_max_abs_cm-1": projector_error,
        "minimum_adjacent_DFPT_top_mode_overlap_squared": continuity,
    }
    (OUTDIR / "qspace_development_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    make_figure(predictions, selected)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
