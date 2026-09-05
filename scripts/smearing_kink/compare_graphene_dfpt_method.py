#!/usr/bin/env python3
"""Compare frozen graphene P0 laws with direct DFPT on identical line-cut q points."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_graphene_dfpt_linecuts as dfpt  # noqa: E402
import friedel_module as fm  # noqa: E402
import graphene_p0_deploy as p0  # noqa: E402


def load_cached_fc(cache_dir: Path, dg: float):
    """Load the exact real-MACE finite-displacement FC archived on the 2060."""
    path = cache_dir / f"graphene_8x8_dg{p0.dg_slug(dg)}_p0_fc2.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        fc = np.asarray(data["fc2"], float)
    if fc.ndim != 4 or not np.isfinite(fc).all():
        raise ValueError(f"invalid cached force constants: {path}")
    return fc


def qpoint_frequencies(ph, fc, t_values):
    # QE uses Cartesian q=t*(1/3,1/sqrt(3)) in units 2pi/a.  The equivalent
    # phonopy reciprocal-fractional coordinate on this ray is (t/3,t/3,0).
    qpoints = np.array([[t / 3.0, t / 3.0, 0.0] for t in t_values])
    ph.force_constants = fc
    ph.run_qpoints(qpoints, with_dynamical_matrices=False)
    return np.sort(np.asarray(ph.get_qpoints_dict()["frequencies"], float), axis=1) * p0.CM


def grouped_method_rows(grouped_dft, ph, fc, dg: float):
    result = {}
    for region in ("G", "K"):
        dft_group = grouped_dft[(dg, 32, region)]
        t = np.array([row["t_GK"] for row in dft_group], float)
        freq = qpoint_frequencies(ph, fc, t)
        rows = []
        for ti, frequencies in zip(t, freq):
            rows.append(
                {
                    "degauss_Ry": dg,
                    "kgrid": 32,
                    "region": region,
                    "t_GK": float(ti),
                    "freqs_cm": frequencies,
                }
            )
        result[region] = rows
    return result


def audit_cached_bands(ph, cache_dir: Path, npz_path: Path) -> float:
    """Ensure exact 2060 FC caches reproduce the archived deployment bands."""
    with np.load(npz_path) as archived:
        dgs = np.asarray(archived["dgs_Ry"], float)
        bands = np.asarray(archived["method_bands_cm"], float)
    max_error = 0.0
    for index, dg in enumerate(dgs):
        fc = load_cached_fc(cache_dir, float(dg))
        rebuilt = np.asarray(p0.band_and_metrics(ph, fc)["bands_cm"], float)
        if rebuilt.shape != bands[index].shape:
            raise ValueError(
                f"archived band shape mismatch at dg={dg:g}: "
                f"{rebuilt.shape} vs {bands[index].shape}"
            )
        max_error = max(max_error, float(np.max(np.abs(rebuilt - bands[index]))))
    return max_error


def metrics_for_method(grouped_dft, grouped_method, dg: float, law: str):
    method_groups = {
        (dg, 32, region): grouped_method[region] for region in ("G", "K")
    }
    dft_metrics = dfpt.set_metrics(grouped_dft, dg, 32)
    method_metrics = dfpt.set_metrics(method_groups, dg, 32)
    all_dft, all_method = [], []
    region_mae = {}
    for region in ("G", "K"):
        _, dft_f6 = dfpt.top_branch(grouped_dft[(dg, 32, region)])
        _, method_f6 = dfpt.top_branch(grouped_method[region])
        all_dft.extend(dft_f6.tolist())
        all_method.extend(method_f6.tolist())
        region_mae[region] = float(np.mean(np.abs(method_f6 - dft_f6)))
    all_dft = np.asarray(all_dft)
    all_method = np.asarray(all_method)
    return {
        "law": law,
        "degauss_Ry": dg,
        "reference_kgrid": 32,
        "linecut_f6_MAE_cm-1": float(np.mean(np.abs(all_method - all_dft))),
        "Gamma_linecut_f6_MAE_cm-1": region_mae["G"],
        "K_linecut_f6_MAE_cm-1": region_mae["K"],
        "Gamma_center_DFT_cm-1": dft_metrics["gamma_frequency_cm-1"],
        "Gamma_center_method_cm-1": method_metrics["gamma_frequency_cm-1"],
        "K_center_DFT_cm-1": dft_metrics["K_frequency_cm-1"],
        "K_center_method_cm-1": method_metrics["K_frequency_cm-1"],
        "K_slope_jump_DFT_THz_per_t": dft_metrics["K_slope_jump_THz_per_t"],
        "K_slope_jump_method_THz_per_t": method_metrics["K_slope_jump_THz_per_t"],
        "K_slope_jump_abs_error_THz_per_t": abs(
            method_metrics["K_slope_jump_THz_per_t"]
            - dft_metrics["K_slope_jump_THz_per_t"]
        ),
    }


def write_outputs(predictions, metrics, audit, metadata, outdir: Path):
    pred_fields = list(predictions[0])
    with (outdir / "graphene_dfpt_method_linecuts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(predictions)
    metric_fields = list(metrics[0])
    with (outdir / "graphene_dfpt_method_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "status": "complete",
        "comparison_reference": "direct DFPT at identical q; k32 is method-matched, k64 is convergence-only",
        "target_leakage": "none: both laws and the MACE backbone were frozen before these DFPT line cuts",
        "archived_cache_vs_band_max_abs_cm-1": audit,
        "law_metadata": metadata,
        "metrics": metrics,
    }
    (outdir / "graphene_dfpt_method_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def make_figure(grouped_dft, method_groups, metrics, outdir: Path):
    laws = ("log_interp", "unified_anchor")
    colors = {"log_interp": "#D55E00", "unified_anchor": "#0072B2"}
    labels = {"log_interp": "log interpolation", "unified_anchor": "unified law"}
    dgs = (0.01, 0.04, 0.08)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0))
    for col, dg in enumerate(dgs):
        for row_index, region in enumerate(("G", "K")):
            ax = axes[row_index, col]
            t, dft_f6 = dfpt.top_branch(grouped_dft[(dg, 32, region)])
            ax.plot(t, dft_f6, "-o", color="#222222", lw=1.5, ms=4,
                    label="direct DFT, 32x32", zorder=4)
            if dg == 0.01:
                t64, dft64_f6 = dfpt.top_branch(grouped_dft[(dg, 64, region)])
                ax.plot(t64, dft64_f6, ":s", color="#888888", lw=1.25, ms=3.5,
                        label="direct DFT, 64x64", zorder=3)
            for law in laws:
                tm, method_f6 = dfpt.top_branch(method_groups[(law, dg, region)])
                ax.plot(
                    tm,
                    method_f6,
                    "--" if law == "log_interp" else "-.",
                    color=colors[law],
                    lw=1.45,
                    label=labels[law],
                    zorder=2,
                )
            if region == "K":
                ax.axvline(1.0, color="#bbbbbb", lw=0.8, zorder=0)
            ax.set_xlabel(r"$t$ along $q=tK$")
            if col == 0:
                ax.set_ylabel(r"highest optical frequency (cm$^{-1}$)")
            region_label = r"$\Gamma$" if region == "G" else "K"
            ax.set_title(f"{region_label} line cut, dg={dg:g} Ry", fontsize=9.5)
            dfpt.style_axis(ax)
    axes[0, 0].legend(frameon=False, fontsize=7.1, loc="best")

    fig.suptitle(
        "Graphene P0: frozen MLIP + long-range laws vs direct DFPT at identical q points",
        fontsize=11.5,
    )
    fig.text(
        0.5,
        0.012,
        "32x32 is the method-matched electron grid; 64x64 at dg=0.01 is a convergence reference.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"graphene_dfpt_vs_method_p0.{suffix}", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir", type=Path, default=ROOT / "results" / "p0_graphene_dfpt"
    )
    parser.add_argument(
        "--log-npz",
        type=Path,
        default=ROOT / "results" / "p0_graphene_prospective" / "graphene_8x8_holdout.npz",
    )
    parser.add_argument(
        "--unified-npz",
        type=Path,
        default=ROOT / "results" / "p0_graphene_prospective_unified" / "graphene_8x8_holdout.npz",
    )
    parser.add_argument(
        "--log-cache",
        type=Path,
        default=ROOT / "results" / "p0_graphene_prospective" / "cache",
    )
    parser.add_argument(
        "--unified-cache",
        type=Path,
        default=ROOT / "results" / "p0_graphene_prospective_unified" / "cache",
    )
    args = parser.parse_args()
    args.indir.mkdir(parents=True, exist_ok=True)
    grouped_dft = dfpt.validate_and_group(
        dfpt.load_rows(
            [args.indir / "dfpt_linecuts_A.csv", args.indir / "dfpt_linecuts_B.csv"]
        )
    )
    law_npz = {"log_interp": args.log_npz, "unified_anchor": args.unified_npz}
    law_cache = {
        "log_interp": args.log_cache,
        "unified_anchor": args.unified_cache,
    }
    cfg = p0.DATASETS["8x8"]
    dft_paths = p0.discover_yamls(cfg)
    ph = fm.load_ph(dft_paths[p0.require_point(dft_paths, cfg.background_dg)])
    method_groups = {}
    predictions = []
    metrics = []
    audit = {}
    metadata = {
        "log_interp": {"interpolation_axis": "log_T"},
        "unified_anchor": json.loads(
            (args.unified_npz.parent / "graphene_8x8_summary.json").read_text()
        ).get("law_metadata", {}),
    }
    for law in ("log_interp", "unified_anchor"):
        audit[law] = audit_cached_bands(ph, law_cache[law], law_npz[law])
        for dg in (0.01, 0.04, 0.08):
            fc = load_cached_fc(law_cache[law], dg)
            grouped = grouped_method_rows(grouped_dft, ph, fc, dg)
            for region in ("G", "K"):
                method_groups[(law, dg, region)] = grouped[region]
                _, dft_f6 = dfpt.top_branch(grouped_dft[(dg, 32, region)])
                t, method_f6 = dfpt.top_branch(grouped[region])
                for ti, dft_value, method_value in zip(t, dft_f6, method_f6):
                    predictions.append(
                        {
                            "law": law,
                            "degauss_Ry": dg,
                            "reference_kgrid": 32,
                            "region": region,
                            "t_GK": float(ti),
                            "DFPT_f6_cm-1": float(dft_value),
                            "method_f6_cm-1": float(method_value),
                            "signed_error_cm-1": float(method_value - dft_value),
                        }
                    )
            metrics.append(metrics_for_method(grouped_dft, grouped, dg, law))
    if any(value > 1e-6 for value in audit.values()):
        raise RuntimeError(f"exact FC cache does not match archived bands: {audit}")
    write_outputs(predictions, metrics, audit, metadata, args.indir)
    make_figure(grouped_dft, method_groups, metrics, args.indir)
    print(f"wrote direct-DFPT method comparison under {args.indir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
