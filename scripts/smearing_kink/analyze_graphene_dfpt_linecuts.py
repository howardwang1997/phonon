#!/usr/bin/env python3
"""Analyze the prespecified P0 graphene DFPT line cuts through Gamma and K."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CM_PER_THz = 33.356
EXPECTED_T = {
    "G": np.array([0.000, 0.015, 0.030, 0.050, 0.080]),
    "K": np.array([0.920, 0.950, 0.970, 0.985, 1.000, 1.015, 1.030, 1.050, 1.080]),
}
EXPECTED_SETS = {(0.01, 64), (0.01, 32), (0.04, 32), (0.08, 32)}


def load_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                freqs = np.array([float(raw[f"f{i}_cm"]) for i in range(1, 7)])
                rows.append(
                    {
                        "lane": raw["lane"],
                        "degauss_Ry": float(raw["degauss_Ry"]),
                        "kgrid": int(raw["kgrid"]),
                        "region": raw["region"],
                        "t_GK": float(raw["t_GK"]),
                        "freqs_cm": freqs,
                    }
                )
    if not rows:
        raise ValueError("no DFPT rows found")
    return rows


def validate_and_group(rows):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = (float(row["degauss_Ry"]), int(row["kgrid"]), str(row["region"]),
               float(row["t_GK"]))
        if key in seen:
            raise ValueError(f"duplicate DFPT point: {key}")
        seen.add(key)
        freqs = np.asarray(row["freqs_cm"], float)
        if freqs.shape != (6,) or not np.isfinite(freqs).all():
            raise ValueError(f"invalid frequencies at {key}")
        grouped[(key[0], key[1], key[2])].append(row)

    observed_sets = {(key[0], key[1]) for key in grouped}
    if observed_sets != EXPECTED_SETS:
        raise ValueError(
            f"expected sets {sorted(EXPECTED_SETS)}, observed {sorted(observed_sets)}"
        )
    for dg, kgrid in EXPECTED_SETS:
        for region, expected in EXPECTED_T.items():
            group = sorted(grouped[(dg, kgrid, region)], key=lambda row: row["t_GK"])
            actual = np.array([row["t_GK"] for row in group], float)
            if not np.allclose(actual, expected, rtol=0.0, atol=1e-10):
                raise ValueError(
                    f"incomplete q grid for dg={dg:g}, k={kgrid}, {region}: {actual}"
                )
            grouped[(dg, kgrid, region)] = group
    return grouped


def top_branch(group):
    t = np.array([row["t_GK"] for row in group], float)
    f6 = np.array([np.asarray(row["freqs_cm"], float)[-1] for row in group])
    return t, f6


def set_metrics(grouped, dg: float, kgrid: int) -> dict[str, float]:
    tg, fg = top_branch(grouped[(dg, kgrid, "G")])
    tk, fk = top_branch(grouped[(dg, kgrid, "K")])
    left = tk < 1.0
    right = tk > 1.0
    left_fit = np.polyfit(tk[left], fk[left], 1)
    right_fit = np.polyfit(tk[right], fk[right], 1)
    k_index = int(np.argmin(np.abs(tk - 1.0)))
    f_at_k = float(fk[k_index])
    local_cusp_depth = 0.5 * (fk[k_index - 1] + fk[k_index + 1]) - f_at_k
    gamma_fit = np.polyfit(tg[tg <= 0.05], fg[tg <= 0.05], 1)
    return {
        "degauss_Ry": dg,
        "kgrid": kgrid,
        "gamma_frequency_cm-1": float(fg[0]),
        "gamma_softening_at_center_vs_t0.08_cm-1": float(fg[-1] - fg[0]),
        "gamma_near_slope_THz_per_t": float(gamma_fit[0] / CM_PER_THz),
        "K_frequency_cm-1": f_at_k,
        "K_left_slope_THz_per_t": float(left_fit[0] / CM_PER_THz),
        "K_right_slope_THz_per_t": float(right_fit[0] / CM_PER_THz),
        "K_slope_jump_THz_per_t": float(abs(right_fit[0] - left_fit[0]) / CM_PER_THz),
        "K_local_cusp_depth_cm-1": float(local_cusp_depth),
    }


def convergence_metrics(grouped) -> dict[str, float]:
    differences = []
    region_values = {}
    for region in ("G", "K"):
        t32, f32 = top_branch(grouped[(0.01, 32, region)])
        t64, f64 = top_branch(grouped[(0.01, 64, region)])
        if not np.allclose(t32, t64):
            raise ValueError(f"k-grid q points differ in region {region}")
        diff = f32 - f64
        differences.extend(diff.tolist())
        region_values[region] = diff
    differences = np.asarray(differences)
    return {
        "dg0.01_k32_minus_k64_RMSE_cm-1": float(np.sqrt(np.mean(differences**2))),
        "dg0.01_k32_minus_k64_MAE_cm-1": float(np.mean(np.abs(differences))),
        "dg0.01_k32_minus_k64_max_abs_cm-1": float(np.max(np.abs(differences))),
        "dg0.01_Gamma_difference_cm-1": float(region_values["G"][0]),
        "dg0.01_K_difference_cm-1": float(region_values["K"][4]),
    }


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def make_figure(grouped, metrics, outdir: Path):
    colors = {0.01: "#222222", 0.04: "#D55E00", 0.08: "#0072B2"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))

    for ax, region, title in (
        (axes[0, 0], "G", r"$\Gamma$ outward line cut"),
        (axes[0, 1], "K", "K crossing line cut"),
    ):
        for dg, kgrid in sorted(EXPECTED_SETS, key=lambda item: (item[0], -item[1])):
            t, f6 = top_branch(grouped[(dg, kgrid, region)])
            label = f"dg={dg:g}, {kgrid}x{kgrid}"
            ax.plot(
                t,
                f6,
                color=colors[dg],
                ls="-" if kgrid == 64 else "--",
                marker="o",
                ms=3.5,
                lw=1.35,
                label=label,
            )
        ax.set_xlabel(r"$t$ along $q=tK$")
        ax.set_ylabel(r"highest optical frequency (cm$^{-1}$)")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=7.2)
        style_axis(ax)
    axes[0, 1].axvline(1.0, color="#aaaaaa", lw=0.8, zorder=0)

    ax = axes[1, 0]
    k32 = [row for row in metrics if row["kgrid"] == 32]
    k32.sort(key=lambda row: row["degauss_Ry"])
    ax.plot(
        [row["degauss_Ry"] for row in k32],
        [row["K_slope_jump_THz_per_t"] for row in k32],
        "-o",
        color="#D55E00",
        lw=1.4,
        ms=5,
        label="32x32",
    )
    k64 = next(row for row in metrics if row["kgrid"] == 64)
    ax.scatter(
        [k64["degauss_Ry"]],
        [k64["K_slope_jump_THz_per_t"]],
        marker="s",
        s=48,
        color="#222222",
        label="64x64 convergence reference",
        zorder=4,
    )
    ax.set_xlabel("Fermi-Dirac degauss (Ry)")
    ax.set_ylabel(r"K slope jump (THz per $t$)")
    ax.set_title("Direct DFPT Kohn-anomaly strength", fontsize=10)
    ax.legend(frameon=False, fontsize=7.3)
    style_axis(ax)

    ax = axes[1, 1]
    labels, diff = [], []
    for region in ("G", "K"):
        t32, f32 = top_branch(grouped[(0.01, 32, region)])
        _, f64 = top_branch(grouped[(0.01, 64, region)])
        labels.extend([f"{region}\n{value:g}" for value in t32])
        diff.extend((f32 - f64).tolist())
    xpos = np.arange(len(diff))
    ax.axhline(0.0, color="#777777", lw=0.8)
    ax.plot(xpos, diff, "o-", color="#0072B2", ms=3.7, lw=1.1)
    ax.axvline(len(EXPECTED_T["G"]) - 0.5, color="#bbbbbb", ls=":", lw=1.0)
    ax.set_xticks(xpos, labels, rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel(r"$\omega_{32}-\omega_{64}$ (cm$^{-1}$)")
    ax.set_title("Low-smearing k-grid sensitivity (dg=0.01)", fontsize=10)
    style_axis(ax)

    fig.suptitle(
        "Graphene P0 direct DFPT: resolved line shapes at a=2.4600 Angstrom",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"graphene_dfpt_linecuts_p0.{suffix}", dpi=200)
    plt.close(fig)


def write_outputs(metrics, convergence, outdir: Path):
    fields = list(metrics[0])
    with (outdir / "graphene_dfpt_linecuts_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "status": "complete",
        "lattice_a_Angstrom": 2.4600,
        "branch_definition": "highest sorted optical branch f6; no eigenvector tracking",
        "slope_fits": {
            "Gamma": "linear fit over t=0,0.015,0.030,0.050",
            "K_left": "linear fit over t=0.920,0.950,0.970,0.985",
            "K_right": "linear fit over t=1.015,1.030,1.050,1.080",
        },
        "sets": metrics,
        "kgrid_convergence": convergence,
    }
    (outdir / "graphene_dfpt_linecuts_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir", type=Path, default=ROOT / "results" / "p0_graphene_dfpt"
    )
    parser.add_argument("--lane-a", type=Path)
    parser.add_argument("--lane-b", type=Path)
    args = parser.parse_args()
    args.indir.mkdir(parents=True, exist_ok=True)
    lane_a = args.lane_a or args.indir / "dfpt_linecuts_A.csv"
    lane_b = args.lane_b or args.indir / "dfpt_linecuts_B.csv"
    grouped = validate_and_group(load_rows([lane_a, lane_b]))
    metrics = [
        set_metrics(grouped, dg, kgrid)
        for dg, kgrid in sorted(EXPECTED_SETS, key=lambda item: (item[0], -item[1]))
    ]
    convergence = convergence_metrics(grouped)
    write_outputs(metrics, convergence, args.indir)
    make_figure(grouped, metrics, args.indir)
    print(f"wrote DFPT P0 analysis under {args.indir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
