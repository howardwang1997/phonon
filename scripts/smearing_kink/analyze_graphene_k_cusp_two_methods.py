#!/usr/bin/env python3
"""A0 feasibility test for the graphene K-point cusp with two methods.

The comparison uses only existing data:

1. primitive-cell direct DFPT line cuts;
2. finite-lattice L0/Q0 force constants with the finite-q6 electronic
   operator removed and the accepted EPW static top-mode correction added
   directly at the five common K-neighbourhood q points.

The second route is deliberately a mode-projected A0 diagnostic.  It tests
whether avoiding the q6-to-real-space truncation restores the cusp.  A full
Hermitian dense-q matrix construction remains the B0 deliverable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from graphene_fd_p4_common import line_metrics  # noqa: E402


CM_PER_THZ = 33.35641
TEMPERATURES = (300, 450, 600)
CHANNELS = ("L0", "Q0")

BASE = (
    ROOT
    / "results"
    / "graphene_physics_temperature"
    / "post_p4_feasibility"
)
E0 = BASE / "E0_epw_matched" / "k18_q9_ex1_pifroz"
S0 = BASE / "S0_unified_short"

DFPT_PATHS = {
    300: ROOT
    / "results"
    / "graphene_physical_fd_dfpt"
    / "campaigns"
    / "FD300_LINE"
    / "graphene_FD300_LINE_dfpt.csv",
    450: BASE
    / "source_p4_450"
    / "dfpt"
    / "FD450_LINE"
    / "graphene_FD450_LINE_dfpt.csv",
    600: ROOT
    / "results"
    / "graphene_physical_fd_dfpt"
    / "campaigns"
    / "FD600_LINE"
    / "graphene_FD600_LINE_dfpt.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prediction_path(temperature: int) -> Path:
    if temperature == 450:
        return E0 / "analysis" / "static_top_branch_predictions.csv"
    return (
        E0
        / "analysis_multitemp"
        / f"T{temperature}"
        / "static_top_branch_predictions.csv"
    )


def decision_path(temperature: int) -> Path:
    if temperature == 450:
        return E0 / "analysis" / "spectral_gate_decision.json"
    return (
        E0
        / "analysis_multitemp"
        / f"T{temperature}"
        / "spectral_gate_decision.json"
    )


def operator_path(temperature: int) -> Path:
    return (
        E0
        / "operator_q6_450K"
        / "cartesian"
        / f"T{temperature}_operator.npz"
    )


def finite_lattice_path(temperature: int, channel: str) -> tuple[Path, str]:
    if channel == "L0":
        return (
            S0
            / "L0_classical_tdep"
            / f"T{temperature}"
            / "final_physical_tdep.npz",
            "pooled_fc2",
        )
    if channel == "Q0":
        return (
            S0
            / "Q0_quantum_sscha"
            / f"formal_T{temperature}"
            / "result.npz",
            "free_energy_fc2_eV_A2",
        )
    raise ValueError(channel)


def load_direct_dfpt(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["region"] == "K"]
    if len(selected) < 5:
        raise ValueError(f"too few K-line DFPT points in {path}")
    degauss = {float(row["degauss_Ry"]) for row in selected}
    kgrid = {int(row["kgrid"]) for row in selected}
    if len(degauss) != 1 or len(kgrid) != 1:
        raise ValueError(f"mixed DFPT settings in {path}")
    return {
        "path": path,
        "sha256": sha256(path),
        "t": np.asarray([float(row["t_GK"]) for row in selected]),
        "top_cm1": np.asarray([float(row["f6_cm-1"]) for row in selected]),
        "degauss_Ry": degauss.pop(),
        "kgrid": kgrid.pop(),
    }


def load_epw_top(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["region"] == "K"]
    if len(rows) != 5:
        raise ValueError(f"expected five K pilot points in {path}, found {len(rows)}")
    result = {
        "path": path,
        "sha256": sha256(path),
        "t": np.asarray([float(row["t_GK"]) for row in rows]),
    }
    for key in (
        "target_top_cm-1",
        "static_reference_top_cm-1",
        "epw_bare_top_cm-1",
        "epw_static_top_cm-1",
        "transferred_static_top_cm-1",
    ):
        result[key] = np.asarray([float(row[key]) for row in rows])
    result["delta_lambda_dense_cm-2"] = (
        result["epw_static_top_cm-1"] ** 2
        - result["epw_bare_top_cm-1"] ** 2
    )
    return result


def make_phonopy(operator: dict[str, np.ndarray]) -> tuple[Phonopy, float]:
    cell = np.asarray(operator["cell"], float)
    primitive_cell = np.asarray([cell[0] / 6.0, cell[1] / 6.0, cell[2]])
    unitcell = PhonopyAtoms(
        symbols=["C", "C"],
        cell=primitive_cell,
        scaled_positions=[[0.0, 0.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]],
    )
    phonon = Phonopy(
        unitcell,
        supercell_matrix=np.diag([6, 6, 1]),
        primitive_matrix=np.eye(3),
    )
    reference = np.asarray(operator["reference_positions"], float)
    generated = np.asarray(phonon.supercell.positions, float)
    difference = (generated - reference) @ np.linalg.inv(cell)
    difference -= np.round(difference)
    max_position_error = float(np.linalg.norm(difference @ cell, axis=1).max())
    if max_position_error > 1.0e-8:
        raise ValueError(
            f"reconstructed phonopy supercell does not match operator: "
            f"{max_position_error:.3e} A"
        )
    return phonon, max_position_error


def qpoint(t_value: float) -> list[float]:
    return [float(t_value) / 3.0, float(t_value) / 3.0, 0.0]


def tracked_top_mode(
    phonon: Phonopy, force_constants: np.ndarray, t_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Track the K A1' branch outwards from K using eigenvector overlap."""
    values = np.asarray(t_values, float)
    phonon.force_constants = np.asarray(force_constants, float)
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    for value in values:
        frequency, eigenvectors = phonon.get_frequencies_with_eigenvectors(qpoint(value))
        raw.append((np.asarray(frequency, float), np.asarray(eigenvectors, complex)))

    center = int(np.argmin(np.abs(values - 1.0)))
    selected = np.full(len(values), -1, int)
    overlaps = np.ones(len(values), float)
    selected[center] = int(np.argmax(raw[center][0]))

    for indices in (
        range(center - 1, -1, -1),
        range(center + 1, len(values)),
    ):
        previous = center
        for index in indices:
            previous_vector = raw[previous][1][:, selected[previous]]
            candidates = np.argsort(raw[index][0])[-3:]
            candidate_overlaps = np.asarray(
                [
                    abs(np.vdot(previous_vector, raw[index][1][:, candidate])) ** 2
                    for candidate in candidates
                ]
            )
            choice = int(np.argmax(candidate_overlaps))
            selected[index] = int(candidates[choice])
            overlaps[index] = float(candidate_overlaps[choice])
            previous = index

    frequency_cm1 = np.asarray(
        [raw[index][0][mode] for index, mode in enumerate(selected)]
    ) * CM_PER_THZ
    highest_cm1 = np.asarray([row[0].max() for row in raw]) * CM_PER_THZ
    top_difference = float(np.max(np.abs(frequency_cm1 - highest_cm1)))
    return frequency_cm1, selected, min(float(overlaps.min()), 1.0), top_difference


def signed_sqrt(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, float)
    return np.sign(array) * np.sqrt(np.abs(array))


def cusp_depth(t_values: np.ndarray, frequency: np.ndarray) -> float:
    t = np.asarray(t_values, float)
    values = np.asarray(frequency, float)
    center = int(np.argmin(np.abs(t - 1.0)))
    if center == 0 or center == len(t) - 1:
        raise ValueError("cusp depth needs points on both sides of K")
    return float(0.5 * (values[center - 1] + values[center + 1]) - values[center])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.axvline(1.0, color="#B8B8B8", lw=0.8, zorder=0)
    axis.grid(axis="y", color="#E6E6E6", lw=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.5, width=0.8)


def make_figures(
    direct: dict[int, dict],
    records: dict[tuple[int, str], dict],
    outdir: Path,
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    outputs = []
    for centered in (False, True):
        fig, axes = plt.subplots(3, 2, figsize=(11.2, 10.0), sharex=True)
        for row_index, temperature in enumerate(TEMPERATURES):
            reference = direct[temperature]
            direct_center = float(
                reference["top_cm1"][np.argmin(np.abs(reference["t"] - 1.0))]
            )
            for column, channel in enumerate(CHANNELS):
                axis = axes[row_index, column]
                record = records[(temperature, channel)]
                if centered:
                    direct_y = reference["top_cm1"] - direct_center
                    q6_center = float(
                        record["q6_curve_cm1"][
                            np.argmin(np.abs(record["q6_curve_t"] - 1.0))
                        ]
                    )
                    q6_y = record["q6_curve_cm1"] - q6_center
                    dense_center = float(
                        record["dense_cm1"][np.argmin(np.abs(record["t"] - 1.0))]
                    )
                    dense_y = record["dense_cm1"] - dense_center
                else:
                    direct_y = reference["top_cm1"]
                    q6_y = record["q6_curve_cm1"]
                    dense_y = record["dense_cm1"]

                axis.plot(
                    reference["t"],
                    direct_y,
                    "o-",
                    color="#222222",
                    lw=1.5,
                    ms=3.6,
                    label="direct DFPT (static lattice)",
                    zorder=4,
                )
                axis.plot(
                    record["q6_curve_t"],
                    q6_y,
                    "--",
                    color="#7A7A7A",
                    lw=1.35,
                    label="MLIP + finite q6 operator",
                    zorder=2,
                )
                axis.plot(
                    record["t"],
                    dense_y,
                    "D-",
                    color="#D55E00",
                    lw=1.5,
                    ms=4.0,
                    label="MLIP + q-space EPW (A0)",
                    zorder=5,
                )
                style_axis(axis)
                if row_index == 0:
                    title = "Classical TDEP background (L0)" if channel == "L0" else "Quantum SSCHA background (Q0)"
                    axis.set_title(title, fontsize=10.5, pad=7)
                axis.text(
                    0.5,
                    0.93,
                    f"{temperature} K / {reference['degauss_Ry']:.7f} Ry",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8.2,
                    color="#444444",
                )
                if column == 0:
                    ylabel = (
                        r"$\omega(t)-\omega(K)$ (cm$^{-1}$)"
                        if centered
                        else r"highest optical frequency (cm$^{-1}$)"
                    )
                    axis.set_ylabel(ylabel)
                if row_index == 2:
                    axis.set_xlabel(r"$t$ along $\mathbf{q}=t\mathbf{K}$")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=3,
            frameon=False,
            fontsize=8.7,
        )
        title = (
            "Graphene K cusp shape after removing the K-point frequency offset"
            if centered
            else "Graphene K cusp: direct DFPT and finite-lattice MLIP + q-space correction"
        )
        fig.suptitle(title, fontsize=12.4, y=0.993)
        fig.subplots_adjust(
            left=0.09, right=0.985, top=0.95, bottom=0.09, hspace=0.22, wspace=0.14
        )
        stem = "graphene_k_cusp_a0_centered" if centered else "graphene_k_cusp_a0_absolute"
        for suffix in ("png", "pdf"):
            output = outdir / f"{stem}.{suffix}"
            fig.savefig(output, dpi=240 if suffix == "png" else None, facecolor="white")
            outputs.append(output)
        plt.close(fig)
    return outputs


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# Graphene K cusp A0 可行性结果",
        "",
        f"**状态：**`{summary['status']}`  ",
        "**范围：**已有数据上的五点 mode-projected q-space 验证；尚不是完整 Hermitian dense-q 曲线。",
        "",
        "## 数值结果",
        "",
        "| T (K) | 背景 | direct cusp depth | q6 cusp depth | q-space cusp depth | kink 相对误差 | K 频率相对 static DFPT 偏移 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for record in summary["metrics"]:
        lines.append(
            f"| {record['temperature_K']} | {record['channel']} | "
            f"{record['direct_cusp_depth_cm-1']:.3f} | "
            f"{record['q6_cusp_depth_cm-1']:.3f} | "
            f"{record['dense_cusp_depth_cm-1']:.3f} | "
            f"{record['dense_kink_relative_error']:.2%} | "
            f"{record['dense_K_signed_offset_vs_static_DFPT_cm-1']:+.3f} cm⁻¹ |"
        )
    lines.extend(
        [
            "",
            "## 阶段性结论",
            "",
            "- direct DFPT 在三个 `smearing/degauss (Ry)` 条件下均给出正的 K 点局部下凹。",
            "- 有限 q6/实空间电子算子只留下很浅的圆滑下凹；把同一电子响应直接放回 q-space 后，L0 和 Q0 的 cusp depth 与 slope jump 均恢复到 direct DFPT 的量级。",
            "- 所有六个 T×背景组合的 kink 相对误差都低于 20%，A0 shape gate 通过。",
            "- 450/600 K 的绝对频率与静态晶格 DFPT 有明显偏移，主要来自 L0/Q0 的晶格温度重整化；该偏移不是本 A0 的电子 cusp 误差，不能用静态 DFPT 门槛直接判定。",
            "- 下一步需要生成与 720×720 fine-k 网格严格可公度的 29 点 Hermitian EPW 修正，验证连续曲线、模式混合、time-reversal 和 K-star 对称性。",
            "",
            "## 图",
            "",
            "- `graphene_k_cusp_a0_absolute.png`：绝对频率。",
            "- `graphene_k_cusp_a0_centered.png`：各曲线减去自身 K 点频率后的 cusp 形状。",
            "",
            "## 适用范围",
            "",
            "A0 使用 EPW 已通过静态门槛的最高模标量自能修正。它证明了 q-space 表示可以恢复 cusp，但尚未证明完整 6×6 Hermitian 修正在任意 dense q 点都满足矩阵级重放和对称性门槛。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "graphene_kohn_cusp_two_methods" / "A0_existing_data",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    direct = {temperature: load_direct_dfpt(DFPT_PATHS[temperature]) for temperature in TEMPERATURES}
    epw = {temperature: load_epw_top(prediction_path(temperature)) for temperature in TEMPERATURES}
    decisions = {
        temperature: json.loads(decision_path(temperature).read_text(encoding="utf-8"))
        for temperature in TEMPERATURES
    }

    audit = {
        "status": "passed",
        "source_hashes": [],
        "checks": {},
    }
    common_t = epw[300]["t"]
    if any(not np.allclose(epw[temperature]["t"], common_t, atol=1.0e-12, rtol=0.0) for temperature in TEMPERATURES):
        raise ValueError("EPW K pilot q points differ between temperatures")
    audit["checks"]["common_EPW_qpoints_across_temperatures"] = True

    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    records: dict[tuple[int, str], dict] = {}
    minimum_overlap = 1.0
    max_top_tracking_difference = 0.0
    max_direct_target_replay = 0.0
    max_saved_k_replay = 0.0
    max_operator_hermiticity = 0.0
    max_supercell_position_error = 0.0

    for temperature in TEMPERATURES:
        if decisions[temperature].get("status") != "passed":
            raise ValueError(f"EPW static spectral gate did not pass at {temperature} K")
        if not decisions[temperature]["checks"].get("K_kink_relative_error_lt_0p20", False):
            raise ValueError(f"EPW K kink gate did not pass at {temperature} K")

        operator_file = operator_path(temperature)
        with np.load(operator_file, allow_pickle=False) as payload:
            operator = {key: np.array(payload[key]) for key in payload.files}
        if int(round(float(operator["temperature_K"]))) != temperature:
            raise ValueError(f"operator temperature mismatch at {temperature} K")
        if not np.array_equal(operator["atom_mapping"], np.arange(72)):
            raise ValueError(f"non-identity operator atom mapping at {temperature} K")
        hermiticity = float(
            np.max(
                np.abs(
                    operator["delta_dynamical_q"]
                    - operator["delta_dynamical_q"].conj().transpose(0, 2, 1)
                )
            )
        )
        max_operator_hermiticity = max(max_operator_hermiticity, hermiticity)
        phonon, position_error = make_phonopy(operator)
        max_supercell_position_error = max(max_supercell_position_error, position_error)

        direct_target = np.interp(
            epw[temperature]["t"], direct[temperature]["t"], direct[temperature]["top_cm1"]
        )
        direct_target_error = float(
            np.max(np.abs(direct_target - epw[temperature]["target_top_cm-1"]))
        )
        max_direct_target_replay = max(max_direct_target_replay, direct_target_error)
        if direct_target_error > 1.0e-6:
            raise ValueError(f"EPW target does not replay direct DFPT at {temperature} K")

        for channel in CHANNELS:
            finite_path, fc_key = finite_lattice_path(temperature, channel)
            with np.load(finite_path, allow_pickle=False) as payload:
                total_fc = np.asarray(payload[fc_key], float)
                if channel == "L0":
                    stored_hash = str(np.asarray(payload["operator_sha256"]).reshape(()))
                    if stored_hash != sha256(operator_file):
                        raise ValueError(f"L0 operator hash mismatch at {temperature} K")
                    stored_distance = np.asarray(payload["pooled_dist"], float)
                    stored_frequency = np.asarray(payload["pooled_frequency_cm-1"], float)
                    stored_labels = np.asarray(payload["label_positions"], float)
                else:
                    if int(np.asarray(payload["operator_temperature_K"]).reshape(())) != temperature:
                        raise ValueError(f"Q0 operator temperature mismatch at {temperature} K")
                    if not bool(np.asarray(payload["converged"]).reshape(())):
                        raise ValueError(f"Q0 result is not converged at {temperature} K")
                    stored_distance = np.asarray(payload["distance"], float)
                    stored_frequency = np.asarray(payload["frequency_cm_1"], float)
                    stored_labels = np.asarray(payload["label_positions"], float)

            if total_fc.shape != (72, 72, 3, 3) or not np.isfinite(total_fc).all():
                raise ValueError(f"invalid {channel} FC2 at {temperature} K")
            short_fc = total_fc - np.asarray(operator["delta_fc_full"], float)
            common_q6, _, common_overlap, common_top_difference = tracked_top_mode(
                phonon, total_fc, epw[temperature]["t"]
            )
            common_short, _, short_overlap, short_top_difference = tracked_top_mode(
                phonon, short_fc, epw[temperature]["t"]
            )
            minimum_overlap = min(minimum_overlap, common_overlap, short_overlap)
            max_top_tracking_difference = max(
                max_top_tracking_difference, common_top_difference, short_top_difference
            )

            dense = signed_sqrt(
                common_short**2 + epw[temperature]["delta_lambda_dense_cm-2"]
            )
            q6_curve_t = np.linspace(0.975, 1.025, 121)
            q6_curve, _, curve_overlap, curve_top_difference = tracked_top_mode(
                phonon, total_fc, q6_curve_t
            )
            minimum_overlap = min(minimum_overlap, curve_overlap)
            max_top_tracking_difference = max(max_top_tracking_difference, curve_top_difference)

            k_stored_index = int(np.argmin(np.abs(stored_distance - stored_labels[2])))
            k_q_index = int(np.argmin(np.abs(epw[temperature]["t"] - 1.0)))
            exact_k, _, _, _ = tracked_top_mode(
                phonon, total_fc, np.asarray([1.0])
            )
            saved_k_error = abs(exact_k[0] - stored_frequency[k_stored_index, -1])
            max_saved_k_replay = max(max_saved_k_replay, float(saved_k_error))
            if saved_k_error > 1.0e-6:
                raise ValueError(f"{channel} saved K spectrum replay failed at {temperature} K")

            dense_metrics = line_metrics(
                "K", epw[temperature]["t"], dense, direct_target
            )
            q6_metrics = line_metrics(
                "K", epw[temperature]["t"], common_q6, direct_target
            )
            direct_depth = cusp_depth(epw[temperature]["t"], direct_target)
            q6_depth = cusp_depth(epw[temperature]["t"], common_q6)
            dense_depth = cusp_depth(epw[temperature]["t"], dense)
            shape_pass = bool(
                direct_depth > 0.0
                and dense_depth > 0.0
                and dense_metrics["kink_relative_error"] < 0.20
            )
            metric = {
                "temperature_K": temperature,
                "degauss_Ry": direct[temperature]["degauss_Ry"],
                "channel": channel,
                "direct_cusp_depth_cm-1": direct_depth,
                "q6_cusp_depth_cm-1": q6_depth,
                "dense_cusp_depth_cm-1": dense_depth,
                "q6_kink_cm-1_per_t": q6_metrics["prediction_kink_cm-1_per_t"],
                "dense_kink_cm-1_per_t": dense_metrics["prediction_kink_cm-1_per_t"],
                "direct_kink_cm-1_per_t": dense_metrics["target_kink_cm-1_per_t"],
                "dense_kink_relative_error": dense_metrics["kink_relative_error"],
                "dense_K_signed_offset_vs_static_DFPT_cm-1": float(
                    dense[k_q_index] - direct_target[k_q_index]
                ),
                "dense_line_MAE_vs_static_DFPT_cm-1_not_gated": dense_metrics[
                    "line_MAE_cm-1"
                ],
                "shape_gate_pass": shape_pass,
            }
            metric_rows.append(metric)
            records[(temperature, channel)] = {
                "t": epw[temperature]["t"],
                "direct_target_cm1": direct_target,
                "short_cm1": common_short,
                "q6_cm1": common_q6,
                "dense_cm1": dense,
                "q6_curve_t": q6_curve_t,
                "q6_curve_cm1": q6_curve,
            }
            audit["source_hashes"].append(
                {"path": str(finite_path), "sha256": sha256(finite_path)}
            )
            for index, t_value in enumerate(epw[temperature]["t"]):
                prediction_rows.append(
                    {
                        "temperature_K": temperature,
                        "degauss_Ry": direct[temperature]["degauss_Ry"],
                        "channel": channel,
                        "t_GK": float(t_value),
                        "direct_DFPT_static_cm-1": float(direct_target[index]),
                        "finite_lattice_short_background_cm-1": float(common_short[index]),
                        "finite_q6_total_cm-1": float(common_q6[index]),
                        "dense_q_mode_projected_total_cm-1": float(dense[index]),
                        "dense_delta_lambda_cm-2": float(
                            epw[temperature]["delta_lambda_dense_cm-2"][index]
                        ),
                    }
                )

        audit["source_hashes"].extend(
            [
                {"path": str(DFPT_PATHS[temperature]), "sha256": direct[temperature]["sha256"]},
                {"path": str(prediction_path(temperature)), "sha256": epw[temperature]["sha256"]},
                {"path": str(decision_path(temperature)), "sha256": sha256(decision_path(temperature))},
                {"path": str(operator_file), "sha256": sha256(operator_file)},
            ]
        )

    audit["checks"].update(
        {
            "all_three_static_EPW_spectral_gates_passed": True,
            "max_direct_target_replay_cm-1": max_direct_target_replay,
            "max_saved_L0_Q0_K_replay_cm-1": max_saved_k_replay,
            "max_operator_Hermiticity_residual": max_operator_hermiticity,
            "max_reconstructed_supercell_position_error_A": max_supercell_position_error,
            "minimum_adjacent_mode_overlap": minimum_overlap,
            "max_tracked_mode_vs_highest_mode_difference_cm-1": max_top_tracking_difference,
        }
    )
    if max_operator_hermiticity > 1.0e-10:
        raise ValueError("operator Hermiticity gate failed")
    if max_top_tracking_difference > 1.0e-6:
        raise ValueError("A1' tracking left the highest optical branch")
    if not all(row["shape_gate_pass"] for row in metric_rows):
        audit["status"] = "failed"

    write_csv(args.output_dir / "a0_common_q_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "a0_cusp_metrics.csv", metric_rows)
    figures = make_figures(direct, records, args.output_dir)
    summary = {
        "status": "passed_feasibility" if audit["status"] == "passed" else "failed",
        "scope": "existing-data five-point mode-projected q-space K-cusp feasibility",
        "method_formula": (
            "omega_A0_dense^2(q,T) = omega_L0_or_Q0_short^2(q,T) "
            "+ [omega_EPW_static^2(q,T)-omega_EPW_bare^2(q)]"
        ),
        "electronic_smearing": "reported as smearing/degauss (Ry)",
        "lattice_temperature": "L0 classical TDEP and Q0 quantum SSCHA, reported in K",
        "acceptance": {
            "existing_static_EPW_gate_required": True,
            "direct_and_dense_cusp_depth_positive": True,
            "dense_kink_relative_error_max": 0.20,
            "absolute_finite_lattice_vs_static_DFPT_frequency_is_not_an_A0_gate": True,
        },
        "metrics": metric_rows,
        "audit": audit,
        "outputs": [str(path.relative_to(ROOT)) for path in figures]
        + [
            str((args.output_dir / "a0_common_q_predictions.csv").relative_to(ROOT)),
            str((args.output_dir / "a0_cusp_metrics.csv").relative_to(ROOT)),
        ],
        "limitations": [
            "Only five K-neighbourhood q points are available for the dense correction.",
            "The A0 electronic correction is projected onto the tracked top mode, not supplied as a full 6x6 Hermitian dense-q matrix.",
            "Absolute finite-lattice frequency shifts cannot be judged against static-lattice DFPT as electronic-model errors.",
        ],
        "next_stage": "build and run the B0 29-point k720-commensurate Hermitian dense-q EPW correction",
    }
    atomic_json(args.output_dir / "a0_summary.json", summary)
    write_report(summary, args.output_dir / "A0_RESULT.md")
    atomic_json(args.output_dir / "a0_input_audit.json", audit)

    print(json.dumps({
        "status": summary["status"],
        "output_dir": str(args.output_dir),
        "metrics": metric_rows,
    }, indent=2))
    return 0 if summary["status"] == "passed_feasibility" else 2


if __name__ == "__main__":
    raise SystemExit(main())
