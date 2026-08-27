#!/usr/bin/env python3
"""Build and gate the B0 dense-q graphene K-cusp comparison.

For each lattice-temperature background, the finite q6 electronic operator is
removed from the accepted L0/Q0 force constants.  The EPW top-mode static
self-energy is then added as a rank-one Hermitian projector onto the tracked
K A1' eigenvector at every dense q point.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spglib


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_graphene_k_cusp_two_methods as a0  # noqa: E402


MEV_TO_CM1 = 8.06554393734921
TEMPERATURES = a0.TEMPERATURES
CHANNELS = a0.CHANNELS


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def display_path(path: Path) -> str:
    """Use repository-relative paths locally and absolute paths for remote /data runs."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_qpoints(path: Path) -> np.ndarray:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    count = int(rows[0][0])
    if rows[0][1] != "crystal" or len(rows) != count + 1:
        raise ValueError(f"invalid dense q-point file: {path}")
    qpoints = np.asarray([[float(value) for value in row[:3]] for row in rows[1:]])
    expected = np.arange(226, 255, dtype=float)[:, None]
    expected_q = np.column_stack(
        (expected[:, 0] / 720.0, expected[:, 0] / 720.0, np.zeros(29))
    )
    if not np.allclose(qpoints, expected_q, atol=5.0e-13, rtol=0.0):
        raise ValueError("dense q points do not match frozen k720-commensurate line")
    return qpoints


def load_self_energy(path: Path, n_qpoints: int, temperature: int) -> dict:
    rows = {}
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 9:
            raise ValueError(f"unexpected specfun row: {line}")
        iq, mode = int(fields[0]), int(fields[1])
        values = [float(value) for value in fields[2:]]
        if abs(values[3]) <= 1.0e-12:
            rows[(iq, mode)] = {
                "temperature_K": values[0],
                "broadening_eV": values[1],
                "bare_eV": values[2],
                "delta_pi_meV": values[4] - values[5],
            }
    expected = {
        (iq, mode)
        for iq in range(1, n_qpoints + 1)
        for mode in range(1, 7)
    }
    if set(rows) != expected:
        raise ValueError(f"incomplete dense self energy in {path}")
    if any(abs(row["temperature_K"] - temperature) > 1.0e-8 for row in rows.values()):
        raise ValueError(f"temperature mismatch in {path}")
    if any(abs(row["broadening_eV"] - 0.005) > 1.0e-12 for row in rows.values()):
        raise ValueError(f"numerical broadening mismatch in {path}")

    bare_cm1 = np.empty((n_qpoints, 6))
    delta_lambda_cm2 = np.empty((n_qpoints, 6))
    for iq in range(1, n_qpoints + 1):
        for mode in range(1, 7):
            row = rows[(iq, mode)]
            bare_mev = row["bare_eV"] * 1000.0
            bare_cm1[iq - 1, mode - 1] = bare_mev * MEV_TO_CM1
            delta_lambda_cm2[iq - 1, mode - 1] = (
                2.0 * bare_mev * row["delta_pi_meV"] * MEV_TO_CM1**2
            )
    return {
        "bare_cm1": bare_cm1,
        "delta_lambda_cm2": delta_lambda_cm2,
        "static_cm1": a0.signed_sqrt(bare_cm1**2 + delta_lambda_cm2),
    }


def dynamical_sequence(phonon, force_constants: np.ndarray, qpoints: np.ndarray) -> dict:
    phonon.force_constants = np.asarray(force_constants, float)
    matrices = []
    eigenvalues = []
    eigenvectors = []
    frequencies = []
    scales = []
    for qpoint in qpoints:
        phonon.dynamical_matrix.run(qpoint)
        matrix = np.asarray(phonon.dynamical_matrix.dynamical_matrix, complex)
        matrix = (matrix + matrix.conj().T) / 2.0
        values, vectors = np.linalg.eigh(matrix)
        frequency = np.asarray(phonon.get_frequencies(qpoint), float) * a0.CM_PER_THZ
        positive = (values > 1.0e-10) & (frequency > 1.0)
        scale = float(np.median(frequency[positive] ** 2 / values[positive]))
        matrices.append(matrix)
        eigenvalues.append(values)
        eigenvectors.append(vectors)
        frequencies.append(frequency)
        scales.append(scale)
    return {
        "matrices": np.asarray(matrices),
        "eigenvalues": np.asarray(eigenvalues),
        "eigenvectors": np.asarray(eigenvectors),
        "frequencies_cm1": np.asarray(frequencies),
        "scale_cm2": np.asarray(scales),
    }


def track_top(eigenvectors: np.ndarray, frequencies: np.ndarray, t_values: np.ndarray) -> tuple[np.ndarray, float]:
    center = int(np.argmin(np.abs(t_values - 1.0)))
    selected = np.full(len(t_values), -1, int)
    overlap = np.ones(len(t_values), float)
    selected[center] = int(np.argmax(frequencies[center]))
    for indices in (range(center - 1, -1, -1), range(center + 1, len(t_values))):
        previous = center
        for index in indices:
            vector = eigenvectors[previous, :, selected[previous]]
            candidates = np.argsort(frequencies[index])[-3:]
            values = np.asarray(
                [abs(np.vdot(vector, eigenvectors[index, :, candidate])) ** 2 for candidate in candidates]
            )
            choice = int(np.argmax(values))
            selected[index] = int(candidates[choice])
            overlap[index] = float(values[choice])
            previous = index
    return selected, float(overlap.min())


def apply_rank_one(
    sequence: dict,
    selected: np.ndarray,
    delta_lambda_top: np.ndarray,
) -> dict:
    corrections = []
    corrected_matrices = []
    corrected_frequencies = []
    scalar_top = []
    hermiticity = 0.0
    scalar_replay = 0.0
    for index, mode in enumerate(selected):
        vector = sequence["eigenvectors"][index, :, mode]
        scale = sequence["scale_cm2"][index]
        correction = (
            float(delta_lambda_top[index]) / scale
        ) * np.outer(vector, vector.conj())
        correction = (correction + correction.conj().T) / 2.0
        corrected = sequence["matrices"][index] + correction
        values = np.linalg.eigvalsh(corrected)
        frequency = a0.signed_sqrt(values * scale)
        expected = a0.signed_sqrt(
            np.asarray(
                [
                    sequence["frequencies_cm1"][index, mode] ** 2
                    + delta_lambda_top[index]
                ]
            )
        )[0]
        corrections.append(correction)
        corrected_matrices.append(corrected)
        corrected_frequencies.append(frequency)
        scalar_top.append(expected)
        hermiticity = max(
            hermiticity,
            float(np.max(np.abs(corrected - corrected.conj().T))),
        )
        scalar_replay = max(scalar_replay, float(abs(frequency[-1] - expected)))
    return {
        "corrections": np.asarray(corrections),
        "corrected_matrices": np.asarray(corrected_matrices),
        "corrected_frequencies_cm1": np.asarray(corrected_frequencies),
        "top_cm1": np.asarray(scalar_top),
        "Hermiticity_max_abs": hermiticity,
        "rank_one_scalar_replay_max_abs_cm1": scalar_replay,
    }


def fixed_cusp_metrics(t_values: np.ndarray, frequency: np.ndarray) -> dict:
    t = np.asarray(t_values, float)
    values = np.asarray(frequency, float)
    center = int(np.argmin(np.abs(t - 1.0)))
    left = int(np.argmin(np.abs(t - (1.0 - 1.0 / 120.0))))
    right = int(np.argmin(np.abs(t - (1.0 + 1.0 / 120.0))))
    delta = float(0.5 * (t[right] - t[left]))
    depth = float(0.5 * (values[left] + values[right]) - values[center])
    slope_jump = float(
        abs(
            (values[right] - values[center]) / delta
            - (values[center] - values[left]) / delta
        )
    )
    return {
        "left_t": float(t[left]),
        "center_t": float(t[center]),
        "right_t": float(t[right]),
        "delta_t": delta,
        "cusp_depth_cm-1": depth,
        "slope_jump_cm-1_per_t": slope_jump,
    }


def reciprocal_star(unitcell, qpoint: np.ndarray) -> list[np.ndarray]:
    symmetry = spglib.get_symmetry(
        (unitcell.cell, unitcell.scaled_positions, unitcell.numbers), symprec=1.0e-7
    )
    if symmetry is None:
        raise RuntimeError("spglib failed to find graphene symmetry")
    result = []
    for rotation in symmetry["rotations"]:
        transformed = np.linalg.inv(rotation).T @ qpoint
        transformed -= np.rint(transformed)
        if not any(np.allclose(transformed, old, atol=1.0e-9, rtol=0.0) for old in result):
            result.append(transformed)
    return result


def star_frequency_spread(
    phonon,
    short_fc: np.ndarray,
    t_values: np.ndarray,
    delta_lambda_top: np.ndarray,
) -> float:
    maximum = 0.0
    for target_t in (1.0 - 1.0 / 120.0, 1.0, 1.0 + 1.0 / 120.0):
        index = int(np.argmin(np.abs(t_values - target_t)))
        q = np.asarray([t_values[index] / 3.0, t_values[index] / 3.0, 0.0])
        frequencies = []
        for star_q in reciprocal_star(phonon.unitcell, q):
            phonon.force_constants = short_fc
            short_top = float(phonon.get_frequencies(star_q)[-1] * a0.CM_PER_THZ)
            frequencies.append(
                float(
                    a0.signed_sqrt(
                        np.asarray([short_top**2 + delta_lambda_top[index]])
                    )[0]
                )
            )
        maximum = max(maximum, float(np.ptp(frequencies)))
    return maximum


def time_reversal_audit(
    phonon,
    short_fc: np.ndarray,
    qpoints: np.ndarray,
    delta_lambda_top: np.ndarray,
) -> dict:
    positive = dynamical_sequence(phonon, short_fc, qpoints)
    negative = dynamical_sequence(phonon, short_fc, -qpoints)
    positive_selected, _ = track_top(
        positive["eigenvectors"], positive["frequencies_cm1"], 3.0 * qpoints[:, 0]
    )
    negative_selected, _ = track_top(
        negative["eigenvectors"], negative["frequencies_cm1"], 3.0 * qpoints[:, 0]
    )
    positive_applied = apply_rank_one(positive, positive_selected, delta_lambda_top)
    negative_applied = apply_rank_one(negative, negative_selected, delta_lambda_top)
    return {
        "short_matrix_max_abs": float(
            np.max(np.abs(negative["matrices"] - positive["matrices"].conj()))
        ),
        "correction_matrix_max_abs": float(
            np.max(
                np.abs(
                    negative_applied["corrections"]
                    - positive_applied["corrections"].conj()
                )
            )
        ),
        "corrected_top_frequency_max_abs_cm-1": float(
            np.max(
                np.abs(
                    negative_applied["top_cm1"] - positive_applied["top_cm1"]
                )
            )
        ),
    }


def pilot_replay(temperature: int, t_values: np.ndarray, dense_static: np.ndarray) -> float:
    pilot = a0.load_epw_top(a0.prediction_path(temperature))
    errors = []
    for pilot_t, pilot_frequency in zip(
        pilot["t"], pilot["epw_static_top_cm-1"], strict=True
    ):
        index = int(np.argmin(np.abs(t_values - pilot_t)))
        if abs(t_values[index] - pilot_t) > 2.0e-5:
            raise ValueError("dense line does not contain all five pilot q points")
        errors.append(abs(dense_static[index, -1] - pilot_frequency))
    return float(max(errors))


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


def make_figures(direct: dict, records: dict, outdir: Path) -> list[Path]:
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
                dense_center = float(
                    record["dense_cm1"][np.argmin(np.abs(record["t"] - 1.0))]
                )
                q6_center = float(
                    record["q6_cm1"][np.argmin(np.abs(record["t"] - 1.0))]
                )
                direct_y = reference["top_cm1"] - direct_center if centered else reference["top_cm1"]
                dense_y = record["dense_cm1"] - dense_center if centered else record["dense_cm1"]
                q6_y = record["q6_cm1"] - q6_center if centered else record["q6_cm1"]
                axis.plot(
                    reference["t"], direct_y, "o-", color="#222222", lw=1.45,
                    ms=3.5, label="direct DFPT (static lattice)", zorder=4,
                )
                axis.plot(
                    record["t"], q6_y, "--", color="#7A7A7A", lw=1.35,
                    label="MLIP + finite q6 operator", zorder=2,
                )
                axis.plot(
                    record["t"], dense_y, "D-", color="#D55E00", lw=1.45,
                    ms=3.0, label="MLIP + Hermitian q-space projector (B0)", zorder=5,
                )
                style_axis(axis)
                if row_index == 0:
                    title = "Classical TDEP background (L0)" if channel == "L0" else "Quantum SSCHA background (Q0)"
                    axis.set_title(title, fontsize=10.5, pad=7)
                axis.text(
                    0.5, 0.93, f"{temperature} K / {reference['degauss_Ry']:.7f} Ry",
                    transform=axis.transAxes, ha="center", va="top", fontsize=8.2,
                    color="#444444",
                )
                if column == 0:
                    axis.set_ylabel(
                        r"$\omega(t)-\omega(K)$ (cm$^{-1}$)"
                        if centered
                        else r"highest optical frequency (cm$^{-1}$)"
                    )
                if row_index == 2:
                    axis.set_xlabel(r"$t$ along $\mathbf{q}=t\mathbf{K}$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.012),
            ncol=3, frameon=False, fontsize=8.5,
        )
        fig.suptitle(
            "Graphene K cusp shape: dense q-space result"
            if centered
            else "Graphene K cusp: direct DFPT and finite-lattice dense q-space method",
            fontsize=12.4, y=0.993,
        )
        fig.subplots_adjust(
            left=0.09, right=0.985, top=0.95, bottom=0.09,
            hspace=0.22, wspace=0.14,
        )
        stem = "graphene_k_cusp_b0_centered" if centered else "graphene_k_cusp_b0_absolute"
        for suffix in ("png", "pdf"):
            output = outdir / f"{stem}.{suffix}"
            fig.savefig(output, dpi=240 if suffix == "png" else None, facecolor="white")
            outputs.append(output)
        plt.close(fig)
    return outputs


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# Graphene K cusp B0 dense-q 结果",
        "",
        f"**状态：**`{summary['status']}`  ",
        "**范围：**29 个与 720×720 fine-k 网格严格可公度的 q 点；MLIP L0/Q0 背景加 K-$A_1'$ Hermitian rank-one EPW 修正。",
        "",
        "## 指标",
        "",
        "| T (K) | 背景 | direct cusp depth | q6 cusp depth | dense cusp depth | kink 相对误差 | K 频率偏移 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["metrics"]:
        lines.append(
            f"| {row['temperature_K']} | {row['channel']} | "
            f"{row['direct_cusp_depth_cm-1']:.3f} | "
            f"{row['q6_cusp_depth_cm-1']:.3f} | "
            f"{row['dense_cusp_depth_cm-1']:.3f} | "
            f"{row['dense_kink_relative_error']:.2%} | "
            f"{row['dense_K_signed_offset_vs_static_DFPT_cm-1']:+.3f} cm⁻¹ |"
        )
    lines.extend(
        [
            "",
            "## 结论与限制",
            "",
            "- dense q-space 结果是否通过，以 cusp depth、slope jump、矩阵 Hermiticity、time-reversal、K-star 和五个旧 pilot 点重放共同判断。",
            "- 绝对频率相对 static-lattice DFPT 的偏移仍包含 lattice-temperature renormalization，不作为电子修正失败的判据。",
            "- 本阶段的 6×6 矩阵是目标 $A_1'$ 模的 Hermitian rank-one 修正；没有给其他五个声子模加入未经验证的电子自能混合。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense-root",
        type=Path,
        default=ROOT
        / "results"
        / "graphene_kohn_cusp_two_methods"
        / "B0_remote_dense_k29",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "graphene_kohn_cusp_two_methods"
        / "B0_dense_k29",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qpoints = load_qpoints(args.dense_root / "qpoints_K29.dat")
    t_values = 3.0 * qpoints[:, 0]
    direct = {temperature: a0.load_direct_dfpt(a0.DFPT_PATHS[temperature]) for temperature in TEMPERATURES}
    self_energy = {}
    remote_audits = {}
    pilot_replay_max = 0.0
    for temperature in TEMPERATURES:
        temperature_root = args.dense_root / f"T{temperature}"
        remote_audits[temperature] = json.loads(
            (temperature_root / "audit.json").read_text(encoding="utf-8")
        )
        if remote_audits[temperature].get("status") != "passed":
            raise ValueError(f"remote dense EPW audit failed at {temperature} K")
        self_energy[temperature] = load_self_energy(
            temperature_root / "specfun_sup.phon", len(qpoints), temperature
        )
        pilot_replay_max = max(
            pilot_replay_max,
            pilot_replay(
                temperature,
                t_values,
                self_energy[temperature]["static_cm1"],
            ),
        )

    records = {}
    metric_rows = []
    prediction_rows = []
    matrix_outputs = []
    global_hermiticity = 0.0
    global_scalar_replay = 0.0
    global_time_reversal_frequency = 0.0
    global_star_spread = 0.0
    global_min_overlap = 1.0

    for temperature in TEMPERATURES:
        operator_file = a0.operator_path(temperature)
        with np.load(operator_file, allow_pickle=False) as payload:
            operator = {key: np.array(payload[key]) for key in payload.files}
        phonon, _ = a0.make_phonopy(operator)
        target = np.interp(
            t_values, direct[temperature]["t"], direct[temperature]["top_cm1"]
        )
        delta_lambda_top = self_energy[temperature]["delta_lambda_cm2"][:, -1]
        for channel in CHANNELS:
            finite_path, fc_key = a0.finite_lattice_path(temperature, channel)
            with np.load(finite_path, allow_pickle=False) as payload:
                total_fc = np.asarray(payload[fc_key], float)
            short_fc = total_fc - np.asarray(operator["delta_fc_full"], float)
            short_sequence = dynamical_sequence(phonon, short_fc, qpoints)
            q6_sequence = dynamical_sequence(phonon, total_fc, qpoints)
            selected, minimum_overlap = track_top(
                short_sequence["eigenvectors"],
                short_sequence["frequencies_cm1"],
                t_values,
            )
            applied = apply_rank_one(short_sequence, selected, delta_lambda_top)
            dense_top = applied["top_cm1"]
            q6_top = q6_sequence["frequencies_cm1"][:, -1]
            direct_cusp = fixed_cusp_metrics(t_values, target)
            q6_cusp = fixed_cusp_metrics(t_values, q6_top)
            dense_cusp = fixed_cusp_metrics(t_values, dense_top)
            kink_relative_error = float(
                abs(
                    dense_cusp["slope_jump_cm-1_per_t"]
                    - direct_cusp["slope_jump_cm-1_per_t"]
                )
                / max(abs(direct_cusp["slope_jump_cm-1_per_t"]), 1.0e-12)
            )
            center = int(np.argmin(np.abs(t_values - 1.0)))
            time_reversal = time_reversal_audit(
                phonon, short_fc, qpoints, delta_lambda_top
            )
            star_spread = star_frequency_spread(
                phonon, short_fc, t_values, delta_lambda_top
            )
            shape_pass = bool(
                direct_cusp["cusp_depth_cm-1"] > 0.0
                and dense_cusp["cusp_depth_cm-1"] > 0.0
                and kink_relative_error < 0.20
                and applied["Hermiticity_max_abs"] <= 1.0e-12
                and applied["rank_one_scalar_replay_max_abs_cm1"] <= 1.0e-6
                and time_reversal["corrected_top_frequency_max_abs_cm-1"] <= 1.0e-6
                and star_spread <= 1.0
                and minimum_overlap >= 0.95
            )
            metric = {
                "temperature_K": temperature,
                "degauss_Ry": direct[temperature]["degauss_Ry"],
                "channel": channel,
                "direct_cusp_depth_cm-1": direct_cusp["cusp_depth_cm-1"],
                "q6_cusp_depth_cm-1": q6_cusp["cusp_depth_cm-1"],
                "dense_cusp_depth_cm-1": dense_cusp["cusp_depth_cm-1"],
                "direct_kink_cm-1_per_t": direct_cusp["slope_jump_cm-1_per_t"],
                "q6_kink_cm-1_per_t": q6_cusp["slope_jump_cm-1_per_t"],
                "dense_kink_cm-1_per_t": dense_cusp["slope_jump_cm-1_per_t"],
                "dense_kink_relative_error": kink_relative_error,
                "dense_K_signed_offset_vs_static_DFPT_cm-1": float(
                    dense_top[center] - target[center]
                ),
                "dense_line_MAE_vs_interpolated_static_DFPT_cm-1_not_gated": float(
                    np.mean(np.abs(dense_top - target))
                ),
                "minimum_adjacent_mode_overlap": minimum_overlap,
                "Hermiticity_max_abs": applied["Hermiticity_max_abs"],
                "rank_one_scalar_replay_max_abs_cm-1": applied[
                    "rank_one_scalar_replay_max_abs_cm1"
                ],
                "time_reversal_top_frequency_max_abs_cm-1": time_reversal[
                    "corrected_top_frequency_max_abs_cm-1"
                ],
                "K_star_top_frequency_spread_max_cm-1": star_spread,
                "shape_and_matrix_gate_pass": shape_pass,
            }
            metric_rows.append(metric)
            records[(temperature, channel)] = {
                "t": t_values,
                "dense_cm1": dense_top,
                "q6_cm1": q6_top,
            }
            for index, t_value in enumerate(t_values):
                prediction_rows.append(
                    {
                        "temperature_K": temperature,
                        "degauss_Ry": direct[temperature]["degauss_Ry"],
                        "channel": channel,
                        "t_GK": float(t_value),
                        "direct_DFPT_static_interpolated_cm-1": float(target[index]),
                        "finite_lattice_short_background_cm-1": float(
                            short_sequence["frequencies_cm1"][index, selected[index]]
                        ),
                        "finite_q6_total_cm-1": float(q6_top[index]),
                        "dense_q_Hermitian_projected_total_cm-1": float(dense_top[index]),
                        "EPW_bare_top_cm-1": float(
                            self_energy[temperature]["bare_cm1"][index, -1]
                        ),
                        "EPW_static_top_cm-1": float(
                            self_energy[temperature]["static_cm1"][index, -1]
                        ),
                        "dense_delta_lambda_top_cm-2": float(delta_lambda_top[index]),
                    }
                )
            matrix_path = args.output_dir / f"T{temperature}_{channel}_dense_matrices.npz"
            atomic_npz(
                matrix_path,
                temperature_K=np.asarray(temperature),
                degauss_Ry=np.asarray(direct[temperature]["degauss_Ry"]),
                channel=np.asarray(channel),
                qpoints_crystal=qpoints,
                t_GK=t_values,
                short_dynamical_matrices=short_sequence["matrices"],
                rank_one_electronic_corrections=applied["corrections"],
                corrected_dynamical_matrices=applied["corrected_matrices"],
                corrected_frequencies_cm1=applied["corrected_frequencies_cm1"],
                tracked_mode_indices=selected,
                tracked_top_frequency_cm1=dense_top,
                matrix_scale_cm2=short_sequence["scale_cm2"],
            )
            matrix_outputs.append(matrix_path)
            global_hermiticity = max(global_hermiticity, applied["Hermiticity_max_abs"])
            global_scalar_replay = max(
                global_scalar_replay,
                applied["rank_one_scalar_replay_max_abs_cm1"],
            )
            global_time_reversal_frequency = max(
                global_time_reversal_frequency,
                time_reversal["corrected_top_frequency_max_abs_cm-1"],
            )
            global_star_spread = max(global_star_spread, star_spread)
            global_min_overlap = min(global_min_overlap, minimum_overlap)

    all_passed = bool(
        all(row["shape_and_matrix_gate_pass"] for row in metric_rows)
        and pilot_replay_max <= 0.2
    )
    write_csv(args.output_dir / "b0_dense_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "b0_dense_metrics.csv", metric_rows)
    figures = make_figures(direct, records, args.output_dir)
    summary = {
        "status": "passed_quantitative_dense" if all_passed else "failed",
        "scope": "29-point k720-commensurate Hermitian rank-one q-space K-cusp validation",
        "q_grid": {
            "n_qpoints": len(qpoints),
            "t_min": float(t_values.min()),
            "t_max": float(t_values.max()),
            "t_step": float(np.median(np.diff(t_values))),
            "commensurability": "q=m/720 and t=m/240 for m=226,...,254",
        },
        "formula": (
            "D_B0(q,T)=D_L0_or_Q0_finite(q,T)-D_E0_q6_to_R(q,T)"
            "+[delta_lambda_EPW_top(q,T)/scale]*|e_A1'(q,T)><e_A1'(q,T)|"
        ),
        "metrics": metric_rows,
        "aggregate_matrix_audit": {
            "Hermiticity_max_abs": global_hermiticity,
            "rank_one_scalar_replay_max_abs_cm-1": global_scalar_replay,
            "time_reversal_top_frequency_max_abs_cm-1": global_time_reversal_frequency,
            "K_star_top_frequency_spread_max_cm-1": global_star_spread,
            "minimum_adjacent_mode_overlap": global_min_overlap,
            "five_pilot_qpoint_EPW_static_replay_max_abs_cm-1": pilot_replay_max,
        },
        "remote_integrity_audits": remote_audits,
        "outputs": [display_path(path) for path in figures + matrix_outputs]
        + [
            display_path(args.output_dir / "b0_dense_predictions.csv"),
            display_path(args.output_dir / "b0_dense_metrics.csv"),
        ],
        "limitations": [
            "The electronic correction is a targeted Hermitian rank-one A1' projector; other modes are left on the finite-lattice short-range background.",
            "Direct DFPT between its eleven computed K-line points is shown by interpolation and is not a new independent DFT calculation.",
            "Absolute finite-lattice frequency offsets relative to static-lattice DFPT are reported but not gated.",
        ],
        "next_stage": "continue the already queued lowest-degauss k192 direct-DFPT densification",
    }
    atomic_json(args.output_dir / "b0_dense_summary.json", summary)
    write_report(summary, args.output_dir / "B0_RESULT.md")
    print(json.dumps({
        "status": summary["status"],
        "output_dir": str(args.output_dir),
        "aggregate_matrix_audit": summary["aggregate_matrix_audit"],
        "metrics": metric_rows,
    }, indent=2))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
