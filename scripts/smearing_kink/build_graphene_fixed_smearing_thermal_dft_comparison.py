#!/usr/bin/env python3
"""Compare E48 at 300 K with the matched-smearing DFT-TDEP reference.

The raw DFT-TDEP force constants contain the electronic response available in
the 6x6 supercell fit.  For a like-for-like near-K comparison, this script also
forms a hybrid DFT reference by removing the same finite-q6 operator used in
E48 and adding the same dense full-EPC response:

    D_DFT-ref = D[Phi_DFT-TDEP - Phi_q6] + Pi_full-EPC |e><e|.

Consequently the hybrid comparison tests the finite-lattice short-range
background; it is not an independent validation of the shared full-EPC term.
"""
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
sys.path.insert(0, str(ROOT / "src"))

import analyze_graphene_k_cusp_b0_dense as b0  # noqa: E402
import analyze_graphene_k_cusp_two_methods as a0  # noqa: E402
import build_graphene_fixed_smearing_thermal_full_epc as e48  # noqa: E402
from phonon_accel.long_range import apply_mode_projected_correction  # noqa: E402


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
TD = ROOT / "results/td_phonon"
TARGET_SMEARING_RY = 0.0019000869
CM_PER_THz = a0.CM_PER_THZ


def scalar(array: np.ndarray):
    return np.asarray(array).reshape(()).item()


def error_metrics(
    coordinate: np.ndarray, mlip: np.ndarray, reference: np.ndarray
) -> dict:
    coordinate = np.asarray(coordinate, float)
    mlip = np.asarray(mlip, float)
    reference = np.asarray(reference, float)
    center = int(np.argmin(np.abs(coordinate)))
    delta = mlip - reference
    shape_delta = (
        mlip - mlip[center] - (reference - reference[center])
    )

    def subset_metrics(mask: np.ndarray) -> dict:
        return {
            "RMSE_cm-1": float(np.sqrt(np.mean(delta[mask] ** 2))),
            "MAE_cm-1": float(np.mean(np.abs(delta[mask]))),
            "max_abs_cm-1": float(np.max(np.abs(delta[mask]))),
            "K_referenced_shape_RMSE_cm-1": float(
                np.sqrt(np.mean(shape_delta[mask] ** 2))
            ),
            "K_referenced_shape_max_abs_cm-1": float(
                np.max(np.abs(shape_delta[mask]))
            ),
        }

    return {
        "sign_convention": "MLIP+full-EPC minus DFT-TDEP+full-EPC",
        "K_error_cm-1": float(delta[center]),
        "full_range": subset_metrics(np.ones(len(coordinate), dtype=bool)),
        "abs_q_le_0.025_2pi_over_a": subset_metrics(
            np.abs(coordinate) <= 0.025 + 1.0e-14
        ),
    }


def load_dft_wave(
    path: Path,
    phonon,
    response: dict,
    delta_fc: np.ndarray,
) -> dict:
    payload = e48.load_npz(path)
    required = {
        "T300_fc2",
        "T300_dist",
        "T300_freq",
        "degauss_Ry",
        "n_structures",
        "fit_rmse_eV_A",
        "primitive_cell",
        "supercell_matrix",
        "label_positions",
    }
    missing = required - set(payload)
    if missing:
        raise KeyError(f"{path} is missing {sorted(missing)}")
    degauss = float(scalar(payload["degauss_Ry"]))
    if abs(degauss - TARGET_SMEARING_RY) > 5.0e-11:
        raise ValueError(f"{path} has degauss={degauss}, expected {TARGET_SMEARING_RY}")
    supercell = np.asarray(payload["supercell_matrix"], float) @ np.asarray(
        payload["primitive_cell"], float
    )
    geometry_error = float(np.max(np.abs(supercell - phonon.supercell.cell)))
    if geometry_error > 1.0e-6:
        raise ValueError(f"{path} supercell differs by {geometry_error:g} A")

    total_fc = np.asarray(payload["T300_fc2"], float)
    if total_fc.shape != delta_fc.shape or not np.isfinite(total_fc).all():
        raise ValueError(f"invalid T300_fc2 in {path}")
    short_fc = total_fc - delta_fc
    reassembly_error = float(np.max(np.abs(short_fc + delta_fc - total_fc)))

    raw_sequence = b0.dynamical_sequence(phonon, total_fc, response["qpoints"])
    raw_indices, raw_overlap, raw_top_error = e48.track_branch(
        raw_sequence, response["signed_distance"]
    )
    raw_frequency = e48.selected_frequency(raw_sequence, raw_indices)

    short_sequence = b0.dynamical_sequence(phonon, short_fc, response["qpoints"])
    short_indices, short_overlap, short_top_error = e48.track_branch(
        short_sequence, response["signed_distance"]
    )
    short_frequency = e48.selected_frequency(short_sequence, short_indices)
    applied = apply_mode_projected_correction(
        short_sequence["matrices"],
        short_sequence["scale_cm2"],
        short_sequence["eigenvectors"],
        short_indices,
        response["full_response_cm2"],
    )
    hybrid_frequency = np.asarray(applied.tracked_frequency_cm1, float)

    center = e48.index_at(response, "K", 0.0)
    stored_distance = np.asarray(payload["T300_dist"], float)
    stored_labels = np.asarray(payload["label_positions"], float)
    stored_frequency = np.asarray(payload["T300_freq"], float) * CM_PER_THz
    stored_k = int(np.argmin(np.abs(stored_distance - stored_labels[2])))
    stored_k_frequency = float(stored_frequency[stored_k, -1])
    k_replay_error = float(abs(raw_frequency[center] - stored_k_frequency))
    if k_replay_error > 1.0e-8:
        raise ValueError(f"{path} raw K replay failed by {k_replay_error:g} cm^-1")

    return {
        "path": path,
        "sha256": e48.sha256(path),
        "n_structures": int(scalar(payload["n_structures"])),
        "fit_RMSE_meV_A": float(scalar(payload["fit_rmse_eV_A"])) * 1000.0,
        "raw_frequency_cm1": raw_frequency,
        "short_frequency_cm1": short_frequency,
        "hybrid_frequency_cm1": hybrid_frequency,
        "raw_K_frequency_cm1": stored_k_frequency,
        "hybrid_K_frequency_cm1": float(hybrid_frequency[center]),
        "geometry_error_A": geometry_error,
        "FC2_reassembly_error_eV_A2": reassembly_error,
        "minimum_mode_overlap": float(min(raw_overlap, short_overlap)),
        "tracked_vs_highest_max_abs_cm1": float(max(raw_top_error, short_top_error)),
        "correction_diagnostics": {
            key: float(value) for key, value in applied.diagnostics.items()
        },
    }


def make_figure(
    coordinate: np.ndarray,
    mlip: np.ndarray,
    waves: dict[int, dict],
    metrics: dict,
    output_png: Path,
    output_pdf: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    center = int(np.argmin(np.abs(coordinate)))
    dft = waves[3]["hybrid_frequency_cm1"]
    raw = waves[3]["raw_frequency_cm1"]
    wave2 = waves[2]["hybrid_frequency_cm1"]
    envelope_low = np.minimum(wave2, dft)
    envelope_high = np.maximum(wave2, dft)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True)
    ax_abs, ax_shape = axes
    ax_abs.fill_between(
        coordinate,
        envelope_low,
        envelope_high,
        color="#999999",
        alpha=0.18,
        linewidth=0,
        label="DFT wave2–wave3 sampling spread",
    )
    ax_abs.plot(
        coordinate,
        mlip,
        color="#2676B8",
        lw=2.2,
        label="MLIP + full-EPC LR",
    )
    ax_abs.plot(
        coordinate,
        dft,
        color="#161616",
        lw=1.9,
        ls="--",
        label="DFT-TDEP (45 labels) + same full-EPC LR",
    )
    ax_abs.plot(
        coordinate,
        raw,
        color="#888888",
        lw=1.15,
        ls=":",
        label="raw 6×6 DFT-TDEP interpolation",
    )
    markers = ("o", "s", "D")
    for marker, wave in zip(markers, (1, 2, 3), strict=True):
        ax_abs.scatter(
            [0.0],
            [waves[wave]["raw_K_frequency_cm1"]],
            s=27,
            marker=marker,
            facecolor="white",
            edgecolor="#555555",
            linewidth=0.9,
            zorder=6,
            label=(
                "raw DFT-TDEP K: 15/30/45 labels" if wave == 1 else None
            ),
        )

    mlip_ref = mlip - mlip[center]
    dft_ref = dft - dft[center]
    wave2_ref = wave2 - wave2[center]
    ax_shape.fill_between(
        coordinate,
        np.minimum(wave2_ref, dft_ref),
        np.maximum(wave2_ref, dft_ref),
        color="#999999",
        alpha=0.18,
        linewidth=0,
    )
    ax_shape.plot(coordinate, mlip_ref, color="#2676B8", lw=2.2)
    ax_shape.plot(coordinate, dft_ref, color="#161616", lw=1.9, ls="--")

    near = metrics["abs_q_le_0.025_2pi_over_a"]
    ax_abs.set_title("(a) Absolute A′ branch at lattice temperature 300 K")
    ax_shape.set_title("(b) K-referenced anomaly shape")
    ax_abs.set_ylabel(r"frequency (cm$^{-1}$)")
    ax_shape.set_ylabel(r"$\omega(q)-\omega(K)$ (cm$^{-1}$)")
    for axis in axes:
        axis.axvline(0.0, color="#B8B8B8", lw=0.8, zorder=0)
        axis.grid(axis="y", color="#E6E6E6", lw=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=3.5, width=0.8)
        axis.set_xlabel(
            r"signed $|q-K|/(2\pi/a)$  (K→Γ < 0; K→M > 0)"
        )
    fig.suptitle(
        "Graphene: matched-smearing finite-lattice DFT comparison\n"
        f"smearing/degauss = {TARGET_SMEARING_RY:.10f} Ry; "
        f"near-K absolute RMSE = {near['RMSE_cm-1']:.2f} cm$^{{-1}}$, "
        f"shape RMSE = {near['K_referenced_shape_RMSE_cm-1']:.2f} cm$^{{-1}}$",
        fontsize=11.8,
        y=0.985,
    )
    handles, labels = ax_abs.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=8.2,
    )
    fig.subplots_adjust(
        left=0.085, right=0.985, top=0.81, bottom=0.29, wspace=0.25
    )
    fig.savefig(output_png, dpi=260, facecolor="white")
    fig.savefig(output_pdf, facecolor="white")
    plt.close(fig)


def write_csv(
    path: Path,
    response: dict,
    mlip: np.ndarray,
    waves: dict[int, dict],
) -> None:
    fields = [
        "smearing_degauss_Ry",
        "lattice_temperature_K",
        "direction",
        "signed_q_2pi_over_a",
        "q1",
        "q2",
        "MLIP_plus_full_EPC_LR_cm-1",
        "DFT_TDEP_wave3_raw_6x6_cm-1",
        "DFT_TDEP_wave1_plus_full_EPC_LR_cm-1",
        "DFT_TDEP_wave2_plus_full_EPC_LR_cm-1",
        "DFT_TDEP_wave3_plus_full_EPC_LR_cm-1",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, qpoint in enumerate(response["qpoints"]):
            writer.writerow(
                {
                    "smearing_degauss_Ry": TARGET_SMEARING_RY,
                    "lattice_temperature_K": 300,
                    "direction": response["direction"][index],
                    "signed_q_2pi_over_a": response["signed_q_2pi_over_a"][index],
                    "q1": qpoint[0],
                    "q2": qpoint[1],
                    "MLIP_plus_full_EPC_LR_cm-1": mlip[index],
                    "DFT_TDEP_wave3_raw_6x6_cm-1": waves[3][
                        "raw_frequency_cm1"
                    ][index],
                    "DFT_TDEP_wave1_plus_full_EPC_LR_cm-1": waves[1][
                        "hybrid_frequency_cm1"
                    ][index],
                    "DFT_TDEP_wave2_plus_full_EPC_LR_cm-1": waves[2][
                        "hybrid_frequency_cm1"
                    ][index],
                    "DFT_TDEP_wave3_plus_full_EPC_LR_cm-1": waves[3][
                        "hybrid_frequency_cm1"
                    ][index],
                }
            )


def write_report(path: Path, summary: dict) -> None:
    metrics = summary["MLIP_vs_DFT_TDEP_plus_shared_full_EPC"]
    near = metrics["abs_q_le_0.025_2pi_over_a"]
    waves = summary["DFT_TDEP_waves"]
    lines = [
        "# Graphene 300 K：MLIP+长程项与 DFT-TDEP 对比",
        "",
        f"状态：`{summary['status']}`",
        "",
        "电子设置统一为 Fermi–Dirac `smearing/degauss = 0.0019000869 Ry`，晶格温度为 300 K。DFT-TDEP 使用累计 15、30、45 个 72 原子构型；主曲线采用 45 标签结果。",
        "",
        "## 数值结果",
        "",
        f"- MLIP+full-EPC 的 K 点频率：`{summary['K_frequencies_cm-1']['MLIP_plus_full_EPC']:.3f} cm^-1`。",
        f"- 45 标签 DFT-TDEP+同一 full-EPC 的 K 点频率：`{summary['K_frequencies_cm-1']['DFT_TDEP_wave3_plus_full_EPC']:.3f} cm^-1`。",
        f"- K 点差值（MLIP−DFT）：`{metrics['K_error_cm-1']:+.3f} cm^-1`。",
        f"- `|q-K|/(2π/a) ≤ 0.025` 内绝对 RMSE：`{near['RMSE_cm-1']:.3f} cm^-1`。",
        f"- 同一区间 K-referenced shape RMSE：`{near['K_referenced_shape_RMSE_cm-1']:.3f} cm^-1`。",
        f"- 原始 DFT-TDEP K 点在 15/30/45 标签时分别为 `{waves['1']['raw_K_frequency_cm1']:.3f}`、`{waves['2']['raw_K_frequency_cm1']:.3f}`、`{waves['3']['raw_K_frequency_cm1']:.3f} cm^-1`。",
        "",
        "## 适用范围",
        "",
        "DFT 混合参考与 MLIP 曲线共享同一个 dense full-EPC 长程响应，因此这张图主要检验有限晶格温度的短程背景和绝对频率，不能作为 full-EPC cusp 形状的独立 DFT 验证。现有 300 K 构型来自旧 cold-smearing DFT-MD 轨迹，再按目标 FD 设置重算力，属于 off-policy 重标注；wave2 到 wave3 仍有可见变化。450 K 同 smearing 重标注完成后，才能形成两温度的完整受控对比。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--response",
        type=Path,
        default=BASE
        / "E42_extended_fixed_lattice_smearing_sweep/extended_fixed_lattice_smearing_sweep.csv",
    )
    parser.add_argument("--operator", type=Path, default=a0.operator_path(300))
    parser.add_argument(
        "--e48-matrices",
        type=Path,
        default=BASE
        / "E48_fixed_smearing_thermal_full_epc/fixed_smearing_thermal_full_epc_matrices.npz",
    )
    parser.add_argument(
        "--dft-root", type=Path, default=TD
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E49_fixed_smearing_thermal_DFT_comparison",
    )
    args = parser.parse_args()

    response = e48.load_response(args.response, TARGET_SMEARING_RY)
    operator = e48.load_npz(args.operator)
    phonon, operator_geometry_error = a0.make_phonopy(operator)
    delta_fc = np.asarray(operator["delta_fc_full"], float)

    e48_payload = e48.load_npz(args.e48_matrices)
    temperatures = np.asarray(e48_payload["lattice_temperature_K"], int)
    matches = np.flatnonzero(temperatures == 300)
    if len(matches) != 1:
        raise ValueError("E48 does not contain exactly one 300 K curve")
    e48_index = int(matches[0])
    mlip = np.asarray(
        e48_payload["MLIP_plus_full_EPC_LR_frequency_cm1"][e48_index], float
    )
    if not np.allclose(
        e48_payload["signed_q_2pi_over_a"],
        response["signed_q_2pi_over_a"],
        atol=1.0e-14,
        rtol=0.0,
    ):
        raise ValueError("E48 and E42 q paths differ")

    waves = {
        wave: load_dft_wave(
            args.dft_root / f"graphene_physical_fd_dft_300K_wave{wave}.npz",
            phonon,
            response,
            delta_fc,
        )
        for wave in (1, 2, 3)
    }
    metrics = error_metrics(
        response["signed_q_2pi_over_a"], mlip, waves[3]["hybrid_frequency_cm1"]
    )
    center = e48.index_at(response, "K", 0.0)
    wave2_to_wave3 = waves[3]["hybrid_frequency_cm1"] - waves[2][
        "hybrid_frequency_cm1"
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "MLIP_full_EPC_vs_DFT_TDEP_300K.png"
    pdf_path = args.output_dir / "MLIP_full_EPC_vs_DFT_TDEP_300K.pdf"
    csv_path = args.output_dir / "MLIP_full_EPC_vs_DFT_TDEP_300K.csv"
    summary_path = args.output_dir / "MLIP_full_EPC_vs_DFT_TDEP_300K_summary.json"
    report_path = args.output_dir / "E49_RESULT.md"
    make_figure(
        response["signed_q_2pi_over_a"],
        mlip,
        waves,
        metrics,
        png_path,
        pdf_path,
    )
    write_csv(csv_path, response, mlip, waves)

    summary = {
        "status": "provisional_300K_matched_smearing_DFT_comparison_450K_pending",
        "scope": "300 K finite-lattice A-prime comparison near graphene K",
        "electronic_control": {
            "smearing": "fermi-dirac",
            "smearing_degauss_Ry": TARGET_SMEARING_RY,
            "same_full_EPC_response_in_MLIP_and_DFT_hybrid_curves": True,
        },
        "lattice_temperature_K": 300,
        "formula_DFT_hybrid": (
            "D[Phi_DFT-TDEP(T=300 K,s)-Phi_q6(s)] "
            "+ Pi_full_EPC(q,s)|e_DFT-short><e_DFT-short|"
        ),
        "K_frequencies_cm-1": {
            "MLIP_plus_full_EPC": float(mlip[center]),
            "DFT_TDEP_wave3_raw_6x6": waves[3]["raw_K_frequency_cm1"],
            "DFT_TDEP_wave3_plus_full_EPC": waves[3]["hybrid_K_frequency_cm1"],
        },
        "MLIP_vs_DFT_TDEP_plus_shared_full_EPC": metrics,
        "DFT_wave2_to_wave3_plus_full_EPC": {
            "K_shift_cm-1": float(wave2_to_wave3[center]),
            "full_range_RMSE_cm-1": float(
                np.sqrt(np.mean(wave2_to_wave3**2))
            ),
            "full_range_max_abs_cm-1": float(np.max(np.abs(wave2_to_wave3))),
        },
        "DFT_TDEP_waves": {
            str(wave): {
                key: str(value) if isinstance(value, Path) else value
                for key, value in record.items()
                if key
                not in {
                    "raw_frequency_cm1",
                    "short_frequency_cm1",
                    "hybrid_frequency_cm1",
                }
            }
            for wave, record in waves.items()
        },
        "checks": {
            "operator_geometry_error_A": operator_geometry_error,
            "all_frequencies_finite": bool(
                np.isfinite(mlip).all()
                and all(
                    np.isfinite(record["hybrid_frequency_cm1"]).all()
                    for record in waves.values()
                )
            ),
            "maximum_DFT_FC2_remove_reassemble_error_eV_A2": max(
                record["FC2_reassembly_error_eV_A2"] for record in waves.values()
            ),
            "minimum_DFT_branch_overlap": min(
                record["minimum_mode_overlap"] for record in waves.values()
            ),
            "maximum_DFT_tracked_vs_highest_abs_cm-1": max(
                record["tracked_vs_highest_max_abs_cm1"] for record in waves.values()
            ),
        },
        "limitations": [
            "the DFT hybrid and MLIP curves share the same full-EPC response, so the cusp is not independently validated here",
            "the 300 K structures were sampled by the earlier cold-smearing DFT-MD trajectory and relabeled at the target FD setting",
            "the cumulative 15/30/45-label waves come from one trajectory and are not independent replicas",
            "450 K target-smearing DFT labels are pending",
        ],
        "provenance": {
            "response": {"path": str(args.response), "sha256": e48.sha256(args.response)},
            "operator": {"path": str(args.operator), "sha256": e48.sha256(args.operator)},
            "E48_matrices": {
                "path": str(args.e48_matrices),
                "sha256": e48.sha256(args.e48_matrices),
            },
        },
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "csv": str(csv_path),
            "report": str(report_path),
        },
    }
    e48.atomic_json(summary_path, summary)
    write_report(report_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
