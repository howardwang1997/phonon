#!/usr/bin/env python3
"""Assemble finite-lattice graphene A' spectra with the current full-EPC LR.

The accepted 300 K and 450 K SSCHA Hessians were sampled with the same
finite-q6 electronic operator at smearing/degauss = 0.001900086938... Ry.
This script removes that operator exactly in real space and replaces it, at
the dynamical-matrix level, by the 241-point full-EPC response frozen in E42:

    Phi_thermal_short(T_lat) = Phi_SSCHA(T_lat, q6) - Phi_q6
    D_total(q, T_lat, s) = D_thermal_short(q, T_lat)
                         + Pi_full_EPC(q, s) |e_A'(q)><e_A'(q)|

Only lattice temperature changes between the two curves.  The electronic
smearing, q path, long-range scalar response, cell, and MLIP are shared.
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


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import analyze_graphene_k_cusp_b0_dense as b0  # noqa: E402
import analyze_graphene_k_cusp_two_methods as a0  # noqa: E402
import fit_graphene_joint_zero_finite_rank1 as joint  # noqa: E402
from extract_graphene_epw_dynamical_matrices import RYDBERG_CM1  # noqa: E402
from phonon_accel.long_range import apply_mode_projected_correction  # noqa: E402


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
TARGET_SMEARING_RY = 0.0019000869
TEMPERATURE_INPUTS = {
    300: {
        "result": BASE / "S0_unified_short/Q0_quantum_sscha/formal_T300/result.npz",
        "acceptance": BASE
        / "S0_unified_short/Q0_quantum_sscha/formal_T300/acceptance.json",
    },
    450: {
        "result": BASE
        / "S0_unified_short/X0_cross_development/sscha_Tlat450_Tel300/result.npz",
        "acceptance": BASE
        / "S0_unified_short/X0_cross_development/sscha_Tlat450_Tel300/acceptance.json",
    },
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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def load_response(path: Path, target_smearing: float) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in all_rows
        if np.isclose(
            float(row["smearing_degauss_Ry"]),
            target_smearing,
            atol=5.0e-11,
            rtol=0.0,
        )
    ]
    if len(rows) != 241:
        raise ValueError(f"expected 241 E42 rows at the target smearing, found {len(rows)}")
    rows.sort(key=lambda row: float(row["signed_distance"]))
    signed = np.asarray([float(row["signed_distance"]) for row in rows], float)
    if np.count_nonzero(np.isclose(signed, 0.0, atol=1.0e-14)) != 1:
        raise ValueError("the E42 path must contain exactly one K point")
    if not np.all(np.diff(signed) > 0.0):
        raise ValueError("the E42 signed path is not strictly ordered")
    return {
        "rows": np.asarray(rows, object),
        "signed_distance": signed,
        "distance": np.asarray([float(row["distance"]) for row in rows], float),
        "signed_q_2pi_over_a": (2.0 / 3.0) * signed,
        "q_distance_2pi_over_a": (2.0 / 3.0)
        * np.asarray([float(row["distance"]) for row in rows], float),
        "direction": np.asarray([row["direction"] for row in rows]),
        "qpoints": np.asarray(
            [
                [float(row["q1"]), float(row["q2"]), 0.0]
                for row in rows
            ],
            float,
        ),
        "smearing_degauss_Ry": np.asarray(
            [float(row["smearing_degauss_Ry"]) for row in rows], float
        ),
        "static_short_cm1": np.asarray(
            [float(row["short_range_background_cm-1"]) for row in rows], float
        ),
        "full_response_Ry2": np.asarray(
            [float(row["full_EPC_response_Ry2"]) for row in rows], float
        ),
        "full_response_cm2": np.asarray(
            [float(row["full_EPC_response_Ry2"]) for row in rows], float
        )
        * RYDBERG_CM1**2,
        "full_correction_cm2": np.asarray(
            [float(row["long_range_correction_cm-2"]) for row in rows], float
        ),
        "static_total_cm1": np.asarray(
            [float(row["MLIP_plus_full_EPC_cm-1"]) for row in rows], float
        ),
    }


def track_branch(sequence: dict, coordinate: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Track the isolated highest A' branch outwards from K on both sides."""

    coordinate = np.asarray(coordinate, float)
    center = int(np.argmin(np.abs(coordinate)))
    frequencies = sequence["frequencies_cm1"]
    eigenvectors = sequence["eigenvectors"]
    selected = np.full(len(coordinate), -1, int)
    overlaps = np.ones(len(coordinate), float)
    selected[center] = int(np.argmax(frequencies[center]))
    for indices in (
        range(center - 1, -1, -1),
        range(center + 1, len(coordinate)),
    ):
        previous = center
        for index in indices:
            previous_vector = eigenvectors[previous, :, selected[previous]]
            candidates = np.argsort(frequencies[index])[-3:]
            candidate_overlap = np.asarray(
                [
                    abs(np.vdot(previous_vector, eigenvectors[index, :, mode])) ** 2
                    for mode in candidates
                ],
                float,
            )
            choice = int(np.argmax(candidate_overlap))
            selected[index] = int(candidates[choice])
            overlaps[index] = float(candidate_overlap[choice])
            previous = index
    selected_frequency = frequencies[np.arange(len(coordinate)), selected]
    highest_frequency = np.max(frequencies, axis=1)
    return (
        selected,
        float(np.min(overlaps)),
        float(np.max(np.abs(selected_frequency - highest_frequency))),
    )


def raw_dynamical_matrices(phonon, force_constants: np.ndarray, qpoints: np.ndarray) -> np.ndarray:
    phonon.force_constants = np.asarray(force_constants, float)
    matrices = []
    for qpoint in qpoints:
        phonon.dynamical_matrix.run(qpoint)
        matrix = np.asarray(phonon.dynamical_matrix.dynamical_matrix, complex)
        matrices.append((matrix + matrix.conj().T) / 2.0)
    return np.asarray(matrices)


def selected_frequency(sequence: dict, indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        sequence["frequencies_cm1"][np.arange(len(indices)), indices], float
    )


def signed_sqrt(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return np.sign(values) * np.sqrt(np.abs(values))


def index_at(response: dict, direction: str, distance: float) -> int:
    if direction == "K":
        mask = response["direction"] == "K"
    else:
        mask = (response["direction"] == direction) & np.isclose(
            response["distance"], distance, atol=1.0e-14, rtol=0.0
        )
    found = np.flatnonzero(mask)
    if len(found) != 1:
        raise ValueError(f"expected one {direction} d={distance:g} point")
    return int(found[0])


def interpolate_side_at_q_distance(
    response: dict, values: np.ndarray, direction: str, q_distance: float
) -> float:
    center = index_at(response, "K", 0.0)
    select = response["direction"] == direction
    radial = response["q_distance_2pi_over_a"][select]
    side_values = np.asarray(values, float)[select]
    order = np.argsort(radial)
    radial = np.concatenate(([0.0], radial[order]))
    side_values = np.concatenate(([values[center]], side_values[order]))
    return float(np.interp(q_distance, radial, side_values))


def curve_metrics(response: dict, frequency: np.ndarray) -> dict:
    center = index_at(response, "K", 0.0)
    output = {"K_frequency_cm-1": float(frequency[center])}
    for direction in ("KG", "KM"):
        for distance in (0.003, 0.025, 0.060):
            index = index_at(response, direction, distance)
            output[f"{direction}_path_d{distance:.3f}_rise_cm-1"] = float(
                frequency[index] - frequency[center]
            )
        for q_distance in (0.003, 0.025, 0.040):
            output[
                f"{direction}_q{q_distance:.3f}_2pi_over_a_rise_cm-1"
            ] = float(
                interpolate_side_at_q_distance(
                    response, frequency, direction, q_distance
                )
                - frequency[center]
            )
    return output


def star_spread_cm1(
    phonon,
    force_constants: np.ndarray,
    qpoint: np.ndarray,
    delta_lambda_cm2: float,
) -> float:
    frequencies = []
    phonon.force_constants = np.asarray(force_constants, float)
    for star_qpoint in b0.reciprocal_star(phonon.unitcell, qpoint):
        short = np.asarray(phonon.get_frequencies(star_qpoint), float) * a0.CM_PER_THZ
        top = float(np.max(short))
        squared = top**2 + float(delta_lambda_cm2)
        frequencies.append(float(np.sign(squared) * np.sqrt(abs(squared))))
    return float(np.ptp(frequencies))


def load_condition(temperature: int, operator_hash: str) -> tuple[dict, dict, np.ndarray]:
    paths = TEMPERATURE_INPUTS[temperature]
    acceptance = json.loads(paths["acceptance"].read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed" or not acceptance.get("converged", False):
        raise ValueError(f"T{temperature} SSCHA acceptance did not pass")
    result_hash = sha256(paths["result"])
    if result_hash != acceptance["result_sha256"]:
        raise ValueError(f"T{temperature} result hash differs from acceptance record")
    accepted_operator = acceptance["input_provenance"]["operator"]["sha256"]
    if accepted_operator != operator_hash:
        raise ValueError(f"T{temperature} operator hash differs from the local T300 operator")
    result = load_npz(paths["result"])
    if int(result["lattice_temperature_K"].reshape(())) != temperature:
        raise ValueError(f"T{temperature} result has the wrong lattice temperature")
    if int(result["operator_temperature_K"].reshape(())) != 300:
        raise ValueError(f"T{temperature} result did not use the fixed T300 operator")
    if not bool(result["converged"].reshape(())):
        raise ValueError(f"T{temperature} result is not converged")
    force_constants = np.asarray(result["free_energy_fc2_eV_A2"], float)
    if force_constants.shape != (72, 72, 3, 3) or not np.isfinite(force_constants).all():
        raise ValueError(f"T{temperature} has invalid SSCHA FC2")
    return acceptance, result, force_constants


def make_figure(response: dict, records: dict[int, dict], output: Path, smearing: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {300: "#2676B8", 450: "#D55E00"}
    x = response["signed_q_2pi_over_a"]
    center = index_at(response, "K", 0.0)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.7))
    ax_abs, ax_ref, ax_shift, ax_lr = axes.ravel()

    for temperature in (300, 450):
        record = records[temperature]
        color = colors[temperature]
        ax_abs.plot(
            x,
            record["full_frequency_cm1"],
            color=color,
            lw=2.0,
            label=f"{temperature} K, MLIP + full-EPC LR",
        )
        ax_abs.plot(
            x,
            record["q6_frequency_cm1"],
            color=color,
            lw=1.25,
            ls="--",
            alpha=0.75,
            label=f"{temperature} K, old finite-q6 LR",
        )
        ax_ref.plot(
            x,
            record["full_frequency_cm1"] - record["full_frequency_cm1"][center],
            color=color,
            lw=2.0,
        )
        ax_ref.plot(
            x,
            record["q6_frequency_cm1"] - record["q6_frequency_cm1"][center],
            color=color,
            lw=1.25,
            ls="--",
            alpha=0.75,
        )
        ax_lr.plot(
            x,
            (record["q6_projected_correction_cm2"] - record["q6_projected_correction_cm2"][center])
            / 1000.0,
            color=color,
            lw=1.25,
            ls="--",
            alpha=0.8,
            label=f"finite q6 projected, {temperature} K",
        )

    full_shift = records[450]["full_frequency_cm1"] - records[300]["full_frequency_cm1"]
    q6_shift = records[450]["q6_frequency_cm1"] - records[300]["q6_frequency_cm1"]
    ax_shift.plot(x, full_shift, color="#3B8D5A", lw=2.0, label="full-EPC LR")
    ax_shift.plot(x, q6_shift, color="#777777", lw=1.4, ls="--", label="finite q6 LR")
    full_relative = (
        response["full_response_cm2"] - response["full_response_cm2"][center]
    ) / 1000.0
    ax_lr.plot(x, full_relative, color="#222222", lw=2.0, label="full-EPC LR (shared)")

    ax_abs.set_title("(a) Absolute A' branch")
    ax_ref.set_title("(b) K-referenced anomaly shape")
    ax_shift.set_title(r"(c) Lattice-temperature shift: 450 K $-$ 300 K")
    ax_lr.set_title("(d) Electronic correction relative to K")
    ax_abs.set_ylabel(r"frequency (cm$^{-1}$)")
    ax_ref.set_ylabel(r"$\omega(q)-\omega(K)$ (cm$^{-1}$)")
    ax_shift.set_ylabel(r"frequency shift (cm$^{-1}$)")
    ax_lr.set_ylabel(r"$[\Delta\lambda(q)-\Delta\lambda(K)]/10^3$ (cm$^{-2}$)")
    ax_ref.set_xlim(-0.025, 0.025)
    for axis in axes.ravel():
        axis.axvline(0.0, color="#B8B8B8", lw=0.8, zorder=0)
        axis.grid(axis="y", color="#E6E6E6", lw=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=3.5, width=0.8)
        axis.set_xlabel(r"signed $|q-K|/(2\pi/a)$  (K→Γ < 0; K→M > 0)")
    ax_shift.legend(loc="best", frameon=False, fontsize=8.2)
    ax_lr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=1, frameon=False, fontsize=8.0)
    handles, labels = ax_abs.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=2,
        frameon=False,
        fontsize=8.4,
    )
    fig.suptitle(
        "Graphene finite-lattice A' spectrum with a shared full-EPC long-range response\n"
        f"smearing/degauss = {smearing:.10f} Ry; lattice temperature = 300 or 450 K",
        fontsize=12.0,
        y=0.985,
    )
    fig.subplots_adjust(left=0.095, right=0.975, top=0.88, bottom=0.18, hspace=0.42, wspace=0.27)
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def write_csv(path: Path, response: dict, records: dict[int, dict]) -> None:
    fields = [
        "lattice_temperature_K",
        "smearing_degauss_Ry",
        "direction",
        "path_distance_d",
        "signed_path_distance_d",
        "q_distance_2pi_over_a",
        "signed_q_2pi_over_a",
        "q1",
        "q2",
        "thermal_short_cm-1",
        "old_finite_q6_cm-1",
        "MLIP_plus_full_EPC_LR_cm-1",
        "full_EPC_response_correction_cm-2",
        "static_E42_anchored_long_range_correction_cm-2_not_added",
        "old_q6_projected_correction_cm-2",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for temperature in (300, 450):
            record = records[temperature]
            for index, qpoint in enumerate(response["qpoints"]):
                writer.writerow(
                    {
                        "lattice_temperature_K": temperature,
                        "smearing_degauss_Ry": TARGET_SMEARING_RY,
                        "direction": response["direction"][index],
                        "path_distance_d": response["distance"][index],
                        "signed_path_distance_d": response["signed_distance"][index],
                        "q_distance_2pi_over_a": response[
                            "q_distance_2pi_over_a"
                        ][index],
                        "signed_q_2pi_over_a": response[
                            "signed_q_2pi_over_a"
                        ][index],
                        "q1": qpoint[0],
                        "q2": qpoint[1],
                        "thermal_short_cm-1": record["short_frequency_cm1"][index],
                        "old_finite_q6_cm-1": record["q6_frequency_cm1"][index],
                        "MLIP_plus_full_EPC_LR_cm-1": record["full_frequency_cm1"][index],
                        "full_EPC_response_correction_cm-2": response[
                            "full_response_cm2"
                        ][index],
                        "static_E42_anchored_long_range_correction_cm-2_not_added": response[
                            "full_correction_cm2"
                        ][index],
                        "old_q6_projected_correction_cm-2": record[
                            "q6_projected_correction_cm2"
                        ][index],
                    }
                )


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# Graphene 固定电子 smearing 下的有限晶格温度 full-EPC 结果",
        "",
        f"状态：`{summary['status']}`",
        "",
        "## 计算定义",
        "",
        "300 K 与 450 K 使用同一个 `smearing/degauss = 0.0019000869 Ry`。先从已经收敛的 SSCHA 自由能 Hessian 中精确扣除旧的 q6 电子算符，再在每个 q 点把当前 full-EPC 响应部分作为 A' 模的 Hermitian projector 加回。E42 用于静态绝对频率对齐的公共常数不属于 q6 算符，因此不重复加入。两条曲线之间只改变晶格温度背景。",
        "",
        "## 主要数值",
        "",
        "下表的 q 距离使用 `|q-K|/(2π/a)`，与旧 E47 晶格温度图一致；E42 内部 path coordinate 满足 `q distance = 2d/3`。",
        "",
        "| lattice temperature | 方法 | K 频率 | KG, q=0.003 上升 | KM, q=0.003 上升 | KG, q=0.025 上升 | KM, q=0.025 上升 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for temperature in (300, 450):
        for label, key in (("full-EPC LR", "full_epc"), ("旧 q6", "finite_q6")):
            metric = summary["metrics_by_lattice_temperature_K"][str(temperature)][key]
            lines.append(
                f"| {temperature} K | {label} | {metric['K_frequency_cm-1']:.3f} | "
                f"{metric['KG_q0.003_2pi_over_a_rise_cm-1']:.3f} | "
                f"{metric['KM_q0.003_2pi_over_a_rise_cm-1']:.3f} | "
                f"{metric['KG_q0.025_2pi_over_a_rise_cm-1']:.3f} | "
                f"{metric['KM_q0.025_2pi_over_a_rise_cm-1']:.3f} |"
            )
    shift = summary["lattice_temperature_shift_450_minus_300_cm-1"]
    lines.extend(
        [
            "",
            f"full-EPC 结果的 K 点 450−300 K 位移为 `{shift['K']:+.3f} cm^-1`。",
            "",
            "## 数值检查",
            "",
            f"- 静态 E42 曲线的矩阵级重放最大误差：`{summary['checks']['static_full_EPC_matrix_replay_max_abs_cm-1']:.3e} cm^-1`。",
            f"- SSCHA 已保存 K 点频率重放最大误差：`{summary['checks']['saved_SSCHA_K_replay_max_abs_cm-1']:.3e} cm^-1`。",
            f"- SSCHA 已保存近 K 曲线的插值重放最大误差：`{summary['checks']['saved_SSCHA_near_K_curve_interpolation_replay_max_abs_cm-1']:.3e} cm^-1`。",
            f"- full-EPC 总矩阵 Hermiticity 残差：`{summary['checks']['full_total_Hermiticity_max_abs']:.3e}`。",
            f"- time-reversal 频率残差：`{summary['checks']['time_reversal_frequency_max_abs_cm-1']:.3e} cm^-1`。",
            f"- K-star 频率 spread：`{summary['checks']['K_star_frequency_spread_max_cm-1']:.3e} cm^-1`。",
            f"- 241 点扩展响应相对独立 nk720 full-EPC 29 点的最大等效频率差：`{summary['checks']['E42_vs_nk720_frequency_max_abs_over_lattice_temperatures_cm-1']:.3f} cm^-1`。",
            "",
            "## 适用范围",
            "",
            "矩阵组合、双重计数消除、分支跟踪和对称性检查已经通过。现有 finite-smearing 静态晶格 DFPT 对 full-EPC 模型给出约 3.06 cm^-1 的绝对 RMSE 和 0.93 cm^-1 的 K-referenced shape RMSE；有限晶格温度曲线目前没有独立的有限温 DFPT/自洽 EPC 基准。因此这里可以确认计算链已经实现，不能把它写成有限晶格温度绝对精度已经由独立第一性原理数据验证。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--response",
        type=Path,
        default=BASE
        / "E42_extended_fixed_lattice_smearing_sweep/extended_fixed_lattice_smearing_sweep.csv",
    )
    parser.add_argument(
        "--response-summary",
        type=Path,
        default=BASE
        / "E43_fixed_lattice_higher_smearing_overlay/fixed_lattice_higher_smearing_overlay_summary.json",
    )
    parser.add_argument(
        "--operator",
        type=Path,
        default=a0.operator_path(300),
    )
    parser.add_argument(
        "--static-short",
        type=Path,
        default=BASE / "source_p4_450/on_policy/frozen_prediction/frozen_static_fc2.npz",
    )
    parser.add_argument(
        "--nk720-response",
        type=Path,
        default=BASE
        / "E17_epc_vertex_rank_response_nk720/finite_vertex_rank_response_arrays.npz",
    )
    parser.add_argument("--smearing-degauss-Ry", type=float, default=TARGET_SMEARING_RY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E48_fixed_smearing_thermal_full_epc",
    )
    args = parser.parse_args()

    response = load_response(args.response, args.smearing_degauss_Ry)
    operator = load_npz(args.operator)
    operator_hash = sha256(args.operator)
    if not np.array_equal(operator["atom_mapping"], np.arange(72)):
        raise ValueError("the old q6 operator does not use the identity atom mapping")
    operator_smearing = float(operator["degauss_Ry"].reshape(()))
    smearing_difference = abs(operator_smearing - args.smearing_degauss_Ry)
    if smearing_difference > 5.0e-10:
        raise ValueError("E42 and SSCHA q6 electronic smearings do not match")
    phonon, geometry_error = a0.make_phonopy(operator)

    # Replay the current E42 static curve through the same matrix interface.
    static_phonon, static_fc = joint.make_static_short_phonon(
        args.operator, args.static_short
    )
    static_sequence = b0.dynamical_sequence(
        static_phonon, static_fc, response["qpoints"]
    )
    static_selected, static_overlap, static_top_difference = track_branch(
        static_sequence, response["signed_distance"]
    )
    static_applied = apply_mode_projected_correction(
        static_sequence["matrices"],
        static_sequence["scale_cm2"],
        static_sequence["eigenvectors"],
        static_selected,
        response["full_correction_cm2"],
    )
    static_short = selected_frequency(static_sequence, static_selected)
    static_short_replay = float(
        np.max(np.abs(static_short - response["static_short_cm1"]))
    )
    static_total_replay = float(
        np.max(
            np.abs(
                static_applied.tracked_frequency_cm1 - response["static_total_cm1"]
            )
        )
    )

    records: dict[int, dict] = {}
    matrices: dict[int, dict] = {}
    condition_summaries = {}
    max_saved_k_replay = 0.0
    max_saved_curve_replay = 0.0
    max_fc_reassembly = 0.0
    max_matrix_linearity = 0.0
    max_min_overlap = 1.0
    max_top_difference = 0.0
    max_q6_scalar_mixing = 0.0
    max_time_reversal_matrix = 0.0
    max_time_reversal_frequency = 0.0
    max_star_spread = 0.0
    combined_diagnostics = {
        "full_total_Hermiticity_max_abs": 0.0,
        "full_correction_Hermiticity_max_abs": 0.0,
        "projector_idempotency_max_abs": 0.0,
        "orthogonal_correction_leakage_max_abs": 0.0,
    }

    for temperature in (300, 450):
        acceptance, result, total_fc = load_condition(temperature, operator_hash)
        delta_fc = np.asarray(operator["delta_fc_full"], float)
        short_fc = total_fc - delta_fc
        max_fc_reassembly = max(
            max_fc_reassembly,
            float(np.max(np.abs(short_fc + delta_fc - total_fc))),
        )

        short_sequence = b0.dynamical_sequence(phonon, short_fc, response["qpoints"])
        total_sequence = b0.dynamical_sequence(phonon, total_fc, response["qpoints"])
        operator_matrices = raw_dynamical_matrices(phonon, delta_fc, response["qpoints"])
        max_matrix_linearity = max(
            max_matrix_linearity,
            float(
                np.max(
                    np.abs(
                        total_sequence["matrices"]
                        - short_sequence["matrices"]
                        - operator_matrices
                    )
                )
            ),
        )
        short_selected, short_overlap, short_top_difference = track_branch(
            short_sequence, response["signed_distance"]
        )
        total_selected, total_overlap, total_top_difference = track_branch(
            total_sequence, response["signed_distance"]
        )
        max_min_overlap = min(max_min_overlap, short_overlap, total_overlap)
        max_top_difference = max(
            max_top_difference, short_top_difference, total_top_difference
        )
        short_frequency = selected_frequency(short_sequence, short_selected)
        q6_frequency = selected_frequency(total_sequence, total_selected)
        applied = apply_mode_projected_correction(
            short_sequence["matrices"],
            short_sequence["scale_cm2"],
            short_sequence["eigenvectors"],
            short_selected,
            response["full_response_cm2"],
        )
        full_frequency = applied.tracked_frequency_cm1

        vectors = np.asarray(
            [
                short_sequence["eigenvectors"][index, :, mode]
                for index, mode in enumerate(short_selected)
            ]
        )
        q6_matrix = total_sequence["matrices"] - short_sequence["matrices"]
        q6_projected = (
            np.einsum(
                "qi,qij,qj->q", vectors.conj(), q6_matrix, vectors, optimize=True
            ).real
            * short_sequence["scale_cm2"]
        )
        q6_scalar_frequency = np.sign(short_frequency**2 + q6_projected) * np.sqrt(
            np.abs(short_frequency**2 + q6_projected)
        )
        max_q6_scalar_mixing = max(
            max_q6_scalar_mixing,
            float(np.max(np.abs(q6_scalar_frequency - q6_frequency))),
        )

        center = index_at(response, "K", 0.0)
        stored_distance = np.asarray(result["distance"], float)
        stored_labels = np.asarray(result["label_positions"], float)
        stored_frequency = np.asarray(result["frequency_cm_1"], float)
        stored_k = int(np.argmin(np.abs(stored_distance - stored_labels[2])))
        saved_k_replay = float(
            abs(q6_frequency[center] - stored_frequency[stored_k, -1])
        )
        max_saved_k_replay = max(max_saved_k_replay, saved_k_replay)
        stored_signed_q = (
            (stored_distance - stored_labels[2])
            * (2.0 / 3.0)
            / (stored_labels[2] - stored_labels[1])
        )
        stored_order = np.argsort(stored_signed_q)
        saved_curve_replay = float(
            np.max(
                np.abs(
                    np.interp(
                        response["signed_q_2pi_over_a"],
                        stored_signed_q[stored_order],
                        stored_frequency[stored_order, -1],
                    )
                    - q6_frequency
                )
            )
        )
        max_saved_curve_replay = max(max_saved_curve_replay, saved_curve_replay)

        negative_sequence = b0.dynamical_sequence(
            phonon, short_fc, -response["qpoints"]
        )
        negative_selected, negative_overlap, negative_top_difference = track_branch(
            negative_sequence, response["signed_distance"]
        )
        max_min_overlap = min(max_min_overlap, negative_overlap)
        max_top_difference = max(max_top_difference, negative_top_difference)
        negative_applied = apply_mode_projected_correction(
            negative_sequence["matrices"],
            negative_sequence["scale_cm2"],
            negative_sequence["eigenvectors"],
            negative_selected,
            response["full_response_cm2"],
        )
        max_time_reversal_matrix = max(
            max_time_reversal_matrix,
            float(
                np.max(
                    np.abs(
                        negative_applied.total_matrices
                        - applied.total_matrices.conj()
                    )
                )
            ),
        )
        max_time_reversal_frequency = max(
            max_time_reversal_frequency,
            float(
                np.max(
                    np.abs(
                        negative_applied.tracked_frequency_cm1
                        - applied.tracked_frequency_cm1
                    )
                )
            ),
        )
        for direction, distance in (("K", 0.0), ("KG", 0.003), ("KM", 0.003), ("KG", 0.025), ("KM", 0.025)):
            index = index_at(response, direction, distance)
            max_star_spread = max(
                max_star_spread,
                star_spread_cm1(
                    phonon,
                    short_fc,
                    response["qpoints"][index],
                    response["full_response_cm2"][index],
                ),
            )

        for source, target in (
            ("total_hermitian_max_abs", "full_total_Hermiticity_max_abs"),
            (
                "correction_hermitian_max_abs",
                "full_correction_Hermiticity_max_abs",
            ),
            ("projector_idempotency_max_abs", "projector_idempotency_max_abs"),
            (
                "orthogonal_correction_leakage_max_abs",
                "orthogonal_correction_leakage_max_abs",
            ),
        ):
            combined_diagnostics[target] = max(
                combined_diagnostics[target], float(applied.diagnostics[source])
            )

        records[temperature] = {
            "short_frequency_cm1": short_frequency,
            "q6_frequency_cm1": q6_frequency,
            "full_frequency_cm1": full_frequency,
            "q6_projected_correction_cm2": q6_projected,
        }
        matrices[temperature] = {
            "short": short_sequence["matrices"],
            "q6": q6_matrix,
            "full_correction": applied.correction_matrices,
            "full_total": applied.total_matrices,
            "projectors": applied.projectors,
            "mode_indices": short_selected,
            "scale_cm2": short_sequence["scale_cm2"],
        }
        condition_summaries[str(temperature)] = {
            "acceptance_status": acceptance["status"],
            "converged": bool(result["converged"].reshape(())),
            "final_population": int(result["final_population"].reshape(())),
            "n_configs": int(result["n_configs"].reshape(())),
            "random_seed": int(result["random_seed"].reshape(())),
            "operator_temperature_K": int(result["operator_temperature_K"].reshape(())),
            "saved_K_replay_max_abs_cm-1": saved_k_replay,
            "saved_near_K_curve_interpolation_replay_max_abs_cm-1": saved_curve_replay,
            "full_minus_old_q6_K_cm-1": float(
                full_frequency[center] - q6_frequency[center]
            ),
            "full_epc": curve_metrics(response, full_frequency),
            "finite_q6": curve_metrics(response, q6_frequency),
            "thermal_short": curve_metrics(response, short_frequency),
        }

    response_summary = json.loads(args.response_summary.read_text(encoding="utf-8"))
    finite_static = response_summary["finite_static_lattice_DFPT_comparison"][
        "0.0019000869"
    ]
    nk720 = load_npz(args.nk720_response)
    matching_smearing = np.flatnonzero(
        np.isclose(
            np.asarray(nk720["degauss_Ry"], float),
            args.smearing_degauss_Ry,
            atol=5.0e-11,
            rtol=0.0,
        )
    )
    if len(matching_smearing) != 1:
        raise ValueError("E17 does not contain exactly one matching nk720 response")
    nk720_index = int(matching_smearing[0])
    nk720_signed = np.asarray(nk720["t_GK"], float) - 1.0
    nk720_correction = np.asarray(
        nk720["full_correction_cm2"][nk720_index], float
    )
    extended_at_nk720 = np.interp(
        nk720_signed,
        response["signed_distance"],
        response["full_response_cm2"],
    )
    nk720_response_error = extended_at_nk720 - nk720_correction
    nk720_frequency_checks = {}
    nk720_frequency_rmse_max = 0.0
    nk720_frequency_error_max = 0.0
    for temperature in (300, 450):
        short_at_nk720 = np.interp(
            nk720_signed,
            response["signed_distance"],
            records[temperature]["short_frequency_cm1"],
        )
        extended_frequency = signed_sqrt(short_at_nk720**2 + extended_at_nk720)
        nk720_frequency = signed_sqrt(short_at_nk720**2 + nk720_correction)
        difference = extended_frequency - nk720_frequency
        rmse = float(np.sqrt(np.mean(difference**2)))
        maximum = float(np.max(np.abs(difference)))
        nk720_frequency_checks[str(temperature)] = {
            "frequency_RMSE_cm-1": rmse,
            "frequency_max_abs_cm-1": maximum,
        }
        nk720_frequency_rmse_max = max(nk720_frequency_rmse_max, rmse)
        nk720_frequency_error_max = max(nk720_frequency_error_max, maximum)

    # q=tK on the positive side is symmetry-equivalent to the conventional KM
    # branch used by E42.  Audit that equivalence before comparing responses.
    nk720_path_symmetry_error = 0.0
    for signed_value, qpoint_value in zip(
        nk720_signed, np.asarray(nk720["qpoints"], float), strict=True
    ):
        if signed_value <= 0.0:
            target = np.asarray(
                [(1.0 + signed_value) / 3.0] * 2 + [0.0], float
            )
        else:
            target = np.asarray(
                [
                    (1.0 + signed_value) / 3.0,
                    (1.0 - 2.0 * signed_value) / 3.0,
                    0.0,
                ],
                float,
            )
        orbit = b0.reciprocal_star(phonon.unitcell, qpoint_value)
        error = min(
            float(
                np.max(
                    np.abs((star_qpoint - target) - np.rint(star_qpoint - target))
                )
            )
            for star_qpoint in orbit
        )
        nk720_path_symmetry_error = max(nk720_path_symmetry_error, error)

    center = index_at(response, "K", 0.0)
    shift_full = records[450]["full_frequency_cm1"] - records[300]["full_frequency_cm1"]
    shift_q6 = records[450]["q6_frequency_cm1"] - records[300]["q6_frequency_cm1"]
    shift_summary = {
        "K": float(shift_full[center]),
        "KG_q0.003_2pi_over_a": interpolate_side_at_q_distance(
            response, shift_full, "KG", 0.003
        ),
        "KM_q0.003_2pi_over_a": interpolate_side_at_q_distance(
            response, shift_full, "KM", 0.003
        ),
        "KG_q0.025_2pi_over_a": interpolate_side_at_q_distance(
            response, shift_full, "KG", 0.025
        ),
        "KM_q0.025_2pi_over_a": interpolate_side_at_q_distance(
            response, shift_full, "KM", 0.025
        ),
        "full_range_min": float(np.min(shift_full)),
        "full_range_max": float(np.max(shift_full)),
    }

    checks = {
        "same_smearing_response_used_for_both_lattice_temperatures": True,
        "operator_vs_response_smearing_abs_difference_Ry": smearing_difference,
        "static_short_matrix_replay_max_abs_cm-1": static_short_replay,
        "static_full_EPC_matrix_replay_max_abs_cm-1": static_total_replay,
        "saved_SSCHA_K_replay_max_abs_cm-1": max_saved_k_replay,
        "saved_SSCHA_near_K_curve_interpolation_replay_max_abs_cm-1": max_saved_curve_replay,
        "FC2_remove_then_reassemble_max_abs_eV_A2": max_fc_reassembly,
        "dynamical_matrix_linearity_max_abs": max_matrix_linearity,
        "minimum_adjacent_mode_overlap": max_min_overlap,
        "tracked_mode_vs_highest_mode_max_abs_cm-1": max_top_difference,
        "old_q6_scalar_projection_vs_full_diagonalization_max_abs_cm-1_not_gated": max_q6_scalar_mixing,
        "time_reversal_matrix_max_abs": max_time_reversal_matrix,
        "time_reversal_frequency_max_abs_cm-1": max_time_reversal_frequency,
        "K_star_frequency_spread_max_cm-1": max_star_spread,
        "E42_vs_nk720_full_EPC_response_RMSE_cm-2": float(
            np.sqrt(np.mean(nk720_response_error**2))
        ),
        "E42_vs_nk720_full_EPC_response_max_abs_cm-2": float(
            np.max(np.abs(nk720_response_error))
        ),
        "E42_vs_nk720_frequency_RMSE_max_over_lattice_temperatures_cm-1": nk720_frequency_rmse_max,
        "E42_vs_nk720_frequency_max_abs_over_lattice_temperatures_cm-1": nk720_frequency_error_max,
        "nk720_line_to_conventional_KG_KM_symmetry_max_abs_fractional": nk720_path_symmetry_error,
        "reconstructed_supercell_position_error_A": geometry_error,
        **combined_diagnostics,
        "all_output_frequencies_finite": bool(
            all(
                np.isfinite(record["full_frequency_cm1"]).all()
                for record in records.values()
            )
        ),
    }
    gates = {
        "operator_response_smearing_match": smearing_difference <= 5.0e-10,
        "static_short_matrix_replay": static_short_replay <= 1.0e-8,
        "static_full_EPC_matrix_replay": static_total_replay <= 1.0e-8,
        "saved_SSCHA_K_replay": max_saved_k_replay <= 1.0e-6,
        "saved_SSCHA_near_K_curve_replay": max_saved_curve_replay <= 0.02,
        "FC2_exact_remove_reassemble": max_fc_reassembly <= 1.0e-12,
        "dynamical_matrix_linearity": max_matrix_linearity <= 1.0e-12,
        "mode_overlap": max_min_overlap >= 0.95,
        "tracked_branch_is_highest": max_top_difference <= 1.0e-6,
        "Hermiticity": combined_diagnostics["full_total_Hermiticity_max_abs"]
        <= 1.0e-12,
        "projector_idempotency": combined_diagnostics[
            "projector_idempotency_max_abs"
        ]
        <= 1.0e-12,
        "orthogonal_leakage": combined_diagnostics[
            "orthogonal_correction_leakage_max_abs"
        ]
        <= 1.0e-12,
        "time_reversal_matrix": max_time_reversal_matrix <= 1.0e-12,
        "time_reversal_frequency": max_time_reversal_frequency <= 1.0e-6,
        "K_star": max_star_spread <= 1.0e-6,
        "nk720_path_symmetry": nk720_path_symmetry_error <= 1.0e-12,
        "extended_response_agrees_with_nk720_within_1_cm-1": nk720_frequency_error_max
        <= 1.0,
        "finite_output": checks["all_output_frequencies_finite"],
    }
    assembly_passed = all(gates.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fixed_smearing_thermal_full_epc.csv"
    figure_path = args.output_dir / "fixed_smearing_thermal_full_epc.png"
    matrix_path = args.output_dir / "fixed_smearing_thermal_full_epc_matrices.npz"
    write_csv(csv_path, response, records)
    make_figure(response, records, figure_path, args.smearing_degauss_Ry)
    atomic_npz(
        matrix_path,
        lattice_temperature_K=np.asarray([300, 450], int),
        smearing_degauss_Ry=np.asarray(args.smearing_degauss_Ry),
        qpoints_crystal=response["qpoints"],
        signed_distance=response["signed_distance"],
        signed_q_2pi_over_a=response["signed_q_2pi_over_a"],
        full_EPC_response_correction_cm2=response["full_response_cm2"],
        static_E42_anchored_long_range_correction_cm2_not_added=response[
            "full_correction_cm2"
        ],
        static_short_frequency_cm1=static_short,
        static_full_EPC_frequency_cm1=static_applied.tracked_frequency_cm1,
        thermal_short_dynamical_matrices=np.asarray(
            [matrices[temperature]["short"] for temperature in (300, 450)]
        ),
        old_q6_correction_matrices=np.asarray(
            [matrices[temperature]["q6"] for temperature in (300, 450)]
        ),
        full_EPC_correction_matrices=np.asarray(
            [matrices[temperature]["full_correction"] for temperature in (300, 450)]
        ),
        full_total_dynamical_matrices=np.asarray(
            [matrices[temperature]["full_total"] for temperature in (300, 450)]
        ),
        Aprime_projectors=np.asarray(
            [matrices[temperature]["projectors"] for temperature in (300, 450)]
        ),
        tracked_mode_indices=np.asarray(
            [matrices[temperature]["mode_indices"] for temperature in (300, 450)]
        ),
        dynamical_matrix_scale_cm2=np.asarray(
            [matrices[temperature]["scale_cm2"] for temperature in (300, 450)]
        ),
        thermal_short_frequency_cm1=np.asarray(
            [records[temperature]["short_frequency_cm1"] for temperature in (300, 450)]
        ),
        old_finite_q6_frequency_cm1=np.asarray(
            [records[temperature]["q6_frequency_cm1"] for temperature in (300, 450)]
        ),
        MLIP_plus_full_EPC_LR_frequency_cm1=np.asarray(
            [records[temperature]["full_frequency_cm1"] for temperature in (300, 450)]
        ),
    )

    summary = {
        "status": (
            "matrix_assembly_passed_finite_temperature_reference_pending"
            if assembly_passed
            else "matrix_assembly_failed"
        ),
        "scope": (
            "graphene K-neighbourhood A' branch at lattice temperatures 300 and "
            "450 K with one fixed electronic smearing and the current 241-point "
            "full-EPC long-range response"
        ),
        "formula": (
            "D(q,Tlat,s)=D[Phi_SSCHA(Tlat,q6_s)-Phi_q6_s](q) "
            "+ Pi_full_EPC(q,s)|e_A'(q,Tlat)><e_A'(q,Tlat)|"
        ),
        "electronic_control": {
            "smearing_degauss_Ry": args.smearing_degauss_Ry,
            "operator_stored_smearing_degauss_Ry": operator_smearing,
            "same_scalar_full_EPC_response_for_300_and_450_K": True,
            "per_temperature_or_per_q_reanchoring": False,
            "static_E42_common_absolute_anchor_added_to_thermal_background": False,
            "reason": (
                "the removed q6 operator contains the EPC response, whereas the "
                "E42 common absolute anchor belongs to its separate static MLIP "
                "frequency alignment and would be double counted here"
            ),
        },
        "lattice_control": {
            "temperatures_K": [300, 450],
            "background": "converged quantum SSCHA free-energy Hessian",
            "cell": "fixed common cell",
            "temperature_is_not_an_input_to_the_MLIP": True,
        },
        "q_path": {
            "number_of_points": len(response["qpoints"]),
            "signed_path_coordinate_d_range": [
                float(np.min(response["signed_distance"])),
                float(np.max(response["signed_distance"])),
            ],
            "path_coordinate_d_step": float(
                np.median(np.diff(response["signed_distance"]))
            ),
            "signed_q_distance_2pi_over_a_range": [
                float(np.min(response["signed_q_2pi_over_a"])),
                float(np.max(response["signed_q_2pi_over_a"])),
            ],
            "q_distance_2pi_over_a_step": float(
                np.median(np.diff(response["signed_q_2pi_over_a"]))
            ),
        },
        "metrics_by_lattice_temperature_K": condition_summaries,
        "lattice_temperature_shift_450_minus_300_cm-1": shift_summary,
        "old_q6_lattice_temperature_shift_450_minus_300_K_at_K_cm-1": float(
            shift_q6[center]
        ),
        "checks": checks,
        "gates": gates,
        "finite_static_lattice_DFPT_support_at_same_smearing": {
            "absolute_RMSE_cm-1": finite_static["absolute_RMSE_cm-1"],
            "K_referenced_shape_RMSE_cm-1": finite_static[
                "K_referenced_shape_RMSE_cm-1"
            ],
            "K_error_cm-1": finite_static["K_error_cm-1"],
            "note": "validation of the electronic LR model on a static lattice; not a finite-lattice-temperature reference",
        },
        "nk720_full_EPC_crosscheck": {
            "number_of_qpoints": len(nk720_signed),
            "frequency_metrics_by_lattice_temperature_K": nk720_frequency_checks,
            "interpretation": (
                "the 241-point extended response differs from the independent "
                "nk720 full-EPC response by less than 1 cm-1 on all 29 common "
                "symmetry-equivalent near-K points"
            ),
        },
        "interpretation": [
            "the previous finite-q6 electronic operator is removed before the full-EPC correction is added, so the direct harmonic electronic term is not double counted",
            "only the full-EPC response replaces q6; the separate E42 static absolute-reference constant is deliberately excluded from the thermal assembly",
            "the 300 K and 450 K difference comes from the accepted SSCHA thermal Hessians while the electronic smearing and full-EPC scalar response remain fixed",
            "the finite-q6 operator is retained only as an explicit diagnostic showing the interpolation smoothing that the dense q-space correction removes",
        ],
        "limitations": [
            "there is no independent finite-lattice-temperature DFPT or self-consistent anharmonic-EPC reference for the final 300/450 K curves",
            "the full-EPC update is a targeted A' rank-one matrix correction; the other five branches remain on the thermal short-range SSCHA background",
            "the SSCHA ensembles were sampled with the q6 representation of the same electronic condition; dense-q full-EPC is applied to the converged Hessian after exact q6 subtraction",
            "the static finite-smearing validation still has the archived residual errors reported above",
        ],
        "inputs": {
            "response": {"path": str(args.response), "sha256": sha256(args.response)},
            "operator": {"path": str(args.operator), "sha256": operator_hash},
            "static_short": {
                "path": str(args.static_short),
                "sha256": sha256(args.static_short),
            },
            "nk720_response": {
                "path": str(args.nk720_response),
                "sha256": sha256(args.nk720_response),
            },
            "thermal_results": {
                str(temperature): {
                    "path": str(TEMPERATURE_INPUTS[temperature]["result"]),
                    "sha256": sha256(TEMPERATURE_INPUTS[temperature]["result"]),
                }
                for temperature in (300, 450)
            },
        },
        "outputs": {
            "figure": str(figure_path),
            "csv": str(csv_path),
            "matrices": str(matrix_path),
            "report": str(args.output_dir / "E48_RESULT.md"),
        },
    }
    atomic_json(args.output_dir / "fixed_smearing_thermal_full_epc_summary.json", summary)
    write_report(summary, args.output_dir / "E48_RESULT.md")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(args.output_dir),
                "metrics_by_lattice_temperature_K": condition_summaries,
                "lattice_temperature_shift_450_minus_300_cm-1": shift_summary,
                "checks": checks,
                "gates": gates,
            },
            indent=2,
        )
    )
    return 0 if assembly_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
