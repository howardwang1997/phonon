#!/usr/bin/env python3
"""Plot the matched-smearing 300/450 K MLIP and DFT-TDEP diagnostics."""
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
import build_graphene_fixed_smearing_thermal_dft_comparison as e49  # noqa: E402
import build_graphene_fixed_smearing_thermal_full_epc as e48  # noqa: E402
from phonon_accel.long_range import apply_mode_projected_correction  # noqa: E402


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
TARGET_SMEARING_RY = 0.0019000869


def assemble(phonon, response: dict, total_fc: np.ndarray, delta_fc: np.ndarray) -> dict:
    total_fc = np.asarray(total_fc, float)
    short_fc = total_fc - delta_fc
    raw_sequence = b0.dynamical_sequence(phonon, total_fc, response["qpoints"])
    raw_indices, raw_overlap, raw_top_error = e48.track_branch(
        raw_sequence, response["signed_distance"]
    )
    raw = e48.selected_frequency(raw_sequence, raw_indices)
    short_sequence = b0.dynamical_sequence(phonon, short_fc, response["qpoints"])
    short_indices, short_overlap, short_top_error = e48.track_branch(
        short_sequence, response["signed_distance"]
    )
    applied = apply_mode_projected_correction(
        short_sequence["matrices"],
        short_sequence["scale_cm2"],
        short_sequence["eigenvectors"],
        short_indices,
        response["full_response_cm2"],
    )
    return {
        "raw_cm1": np.asarray(raw, float),
        "hybrid_cm1": np.asarray(applied.tracked_frequency_cm1, float),
        "minimum_overlap": float(min(raw_overlap, short_overlap)),
        "tracked_vs_highest_max_abs_cm1": float(max(raw_top_error, short_top_error)),
        "FC2_reassembly_max_abs_eV_A2": float(
            np.max(np.abs(short_fc + delta_fc - total_fc))
        ),
        "correction_diagnostics": {
            key: float(value) for key, value in applied.diagnostics.items()
        },
    }


def make_figure(
    x: np.ndarray,
    mlip: dict[int, np.ndarray],
    dft: dict[int, dict],
    variants: dict[int, list[dict]],
    raw_k_markers: dict[int, np.ndarray],
    metrics: dict[int, dict],
    png: Path,
    pdf: Path,
) -> None:
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
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    center = int(np.argmin(np.abs(x)))
    for column, temperature in enumerate((300, 450)):
        ax_abs = axes[0, column]
        ax_shape = axes[1, column]
        variant_curves = np.asarray(
            [record["hybrid_cm1"] for record in variants[temperature]]
        )
        ax_abs.fill_between(
            x,
            np.min(variant_curves, axis=0),
            np.max(variant_curves, axis=0),
            color="#9A9A9A",
            alpha=0.18,
            linewidth=0,
            label="DFT sampling/seed spread",
        )
        ax_abs.plot(
            x,
            mlip[temperature],
            color=colors[temperature],
            lw=2.2,
            label="MLIP + full-EPC LR",
        )
        ax_abs.plot(
            x,
            dft[temperature]["hybrid_cm1"],
            color="#171717",
            lw=1.9,
            ls="--",
            label="DFT-TDEP + same full-EPC LR",
        )
        ax_abs.plot(
            x,
            dft[temperature]["raw_cm1"],
            color="#888888",
            lw=1.15,
            ls=":",
            label="raw 6×6 DFT-TDEP interpolation",
        )
        marker_shapes = ("o", "s", "D")
        for index, value in enumerate(raw_k_markers[temperature]):
            ax_abs.scatter(
                [0.0],
                [value],
                marker=marker_shapes[index],
                s=27,
                facecolor="white",
                edgecolor="#555555",
                linewidth=0.9,
                zorder=6,
                label=("DFT subset K values" if index == 0 else None),
            )

        mlip_ref = mlip[temperature] - mlip[temperature][center]
        dft_ref = dft[temperature]["hybrid_cm1"] - dft[temperature][
            "hybrid_cm1"
        ][center]
        variant_ref = variant_curves - variant_curves[:, center, None]
        ax_shape.fill_between(
            x,
            np.min(variant_ref, axis=0),
            np.max(variant_ref, axis=0),
            color="#9A9A9A",
            alpha=0.18,
            linewidth=0,
        )
        ax_shape.plot(x, mlip_ref, color=colors[temperature], lw=2.2)
        ax_shape.plot(x, dft_ref, color="#171717", lw=1.9, ls="--")
        near = metrics[temperature]["abs_q_le_0.025_2pi_over_a"]
        ax_abs.set_title(
            f"({chr(97 + column)}) Absolute A′, lattice temperature {temperature} K\n"
            f"K error (MLIP−DFT) = {metrics[temperature]['K_error_cm-1']:+.2f} cm$^{{-1}}$"
        )
        ax_shape.set_title(
            f"({chr(99 + column)}) K-referenced shape, {temperature} K\n"
            f"near-K shape RMSE = {near['K_referenced_shape_RMSE_cm-1']:.2f} cm$^{{-1}}$"
        )
        ax_abs.set_ylabel(r"frequency (cm$^{-1}$)")
        ax_shape.set_ylabel(r"$\omega(q)-\omega(K)$ (cm$^{-1}$)")

    for axis in axes.ravel():
        axis.axvline(0.0, color="#B8B8B8", lw=0.8, zorder=0)
        axis.grid(axis="y", color="#E6E6E6", lw=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=3.5, width=0.8)
        axis.set_xlabel(r"signed $|q-K|/(2\pi/a)$  (K→Γ < 0; K→M > 0)")
        axis.set_xlim(-0.042, 0.042)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=8.3,
    )
    fig.suptitle(
        "Graphene matched-smearing finite-lattice DFT diagnostic\n"
        "smearing/degauss = 0.0019000869 Ry; 450 K DFT data are off-policy",
        fontsize=12.0,
        y=0.988,
    )
    fig.subplots_adjust(
        left=0.085, right=0.985, top=0.87, bottom=0.16, hspace=0.39, wspace=0.25
    )
    fig.savefig(png, dpi=260, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)


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
        "--e48",
        type=Path,
        default=BASE
        / "E48_fixed_smearing_thermal_full_epc/fixed_smearing_thermal_full_epc_matrices.npz",
    )
    parser.add_argument(
        "--dft300-root", type=Path, default=ROOT / "results/td_phonon"
    )
    parser.add_argument(
        "--dft450",
        type=Path,
        default=BASE
        / "E50_fixed_smearing_T450_DFT_reference/graphene_fixed_smearing_DFT_TDEP_450K.npz",
    )
    parser.add_argument(
        "--old-dft450",
        type=Path,
        default=BASE / "source_p4_450/on_policy/dft_tdep_60.npz",
    )
    parser.add_argument(
        "--old-p4-acceptance",
        type=Path,
        default=BASE
        / "source_p4_450/on_policy/evaluation/transferability_acceptance.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E51_fixed_smearing_thermal_DFT_pair",
    )
    args = parser.parse_args()

    response = e48.load_response(args.response, TARGET_SMEARING_RY)
    operator = e48.load_npz(args.operator)
    phonon, geometry_error = a0.make_phonopy(operator)
    delta_fc = np.asarray(operator["delta_fc_full"], float)
    e48_data = e48.load_npz(args.e48)
    temperatures = np.asarray(e48_data["lattice_temperature_K"], int)
    mlip = {
        temperature: np.asarray(
            e48_data["MLIP_plus_full_EPC_LR_frequency_cm1"][
                int(np.flatnonzero(temperatures == temperature)[0])
            ],
            float,
        )
        for temperature in (300, 450)
    }

    dft300_payloads = {
        wave: e48.load_npz(
            args.dft300_root / f"graphene_physical_fd_dft_300K_wave{wave}.npz"
        )
        for wave in (1, 2, 3)
    }
    dft300 = assemble(
        phonon, response, dft300_payloads[3]["T300_fc2"], delta_fc
    )
    dft300_variants = [
        assemble(phonon, response, dft300_payloads[wave]["T300_fc2"], delta_fc)
        for wave in (2, 3)
    ]
    dft450_payload = e48.load_npz(args.dft450)
    if abs(float(dft450_payload["degauss_Ry"].reshape(())) - TARGET_SMEARING_RY) > 5e-11:
        raise ValueError("450 K DFT-TDEP has the wrong smearing")
    dft450 = assemble(phonon, response, dft450_payload["T450_fc2"], delta_fc)
    dft450_variants = [
        assemble(phonon, response, fc2, delta_fc)
        for fc2 in dft450_payload["leave_one_seed_out_fc2"]
    ]
    dft = {300: dft300, 450: dft450}
    variants = {300: dft300_variants, 450: dft450_variants}
    x = response["signed_q_2pi_over_a"]
    metrics = {
        temperature: e49.error_metrics(x, mlip[temperature], dft[temperature]["hybrid_cm1"])
        for temperature in (300, 450)
    }
    center = e48.index_at(response, "K", 0.0)

    old450 = e48.load_npz(args.old_dft450)
    old_k_index = int(
        np.argmin(np.abs(old450["T450_dist"] - old450["label_positions"][2]))
    )
    old450_k = float(old450["T450_freq"][old_k_index, -1] * a0.CM_PER_THZ)
    new450_raw_k = float(dft450["raw_cm1"][center])
    old_acceptance = json.loads(args.old_p4_acceptance.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "MLIP_full_EPC_vs_DFT_TDEP_300K_450K.png"
    pdf = args.output_dir / "MLIP_full_EPC_vs_DFT_TDEP_300K_450K.pdf"
    csv_path = args.output_dir / "MLIP_full_EPC_vs_DFT_TDEP_300K_450K.csv"
    summary_path = args.output_dir / "MLIP_full_EPC_vs_DFT_TDEP_300K_450K_summary.json"
    report_path = args.output_dir / "E51_RESULT.md"
    raw_k_markers = {
        300: np.asarray(
            [
                float(
                    payload["T300_freq"][
                        int(
                            np.argmin(
                                np.abs(payload["T300_dist"] - payload["label_positions"][2])
                            )
                        ),
                        -1,
                    ]
                    * a0.CM_PER_THZ
                )
                for payload in dft300_payloads.values()
            ]
        ),
        450: np.asarray(dft450_payload["per_seed_K_top_cm_1"], float),
    }
    make_figure(x, mlip, dft, variants, raw_k_markers, metrics, png, pdf)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "lattice_temperature_K",
                "signed_q_2pi_over_a",
                "direction",
                "MLIP_plus_full_EPC_cm-1",
                "DFT_TDEP_plus_same_full_EPC_cm-1",
                "raw_DFT_TDEP_cm-1",
            ]
        )
        for temperature in (300, 450):
            for index in range(len(x)):
                writer.writerow(
                    [
                        temperature,
                        x[index],
                        response["direction"][index],
                        mlip[temperature][index],
                        dft[temperature]["hybrid_cm1"][index],
                        dft[temperature]["raw_cm1"][index],
                    ]
                )

    summary = {
        "status": "450K_off_policy_DFT_diagnostic_failed_as_thermodynamic_validation",
        "electronic_control": {
            "smearing": "fermi-dirac",
            "degauss_Ry": TARGET_SMEARING_RY,
            "same_dense_full_EPC_response_for_MLIP_and_DFT_hybrid": True,
        },
        "metrics_by_lattice_temperature_K": {
            str(temperature): {
                "MLIP_K_cm-1": float(mlip[temperature][center]),
                "DFT_TDEP_raw_K_cm-1": float(dft[temperature]["raw_cm1"][center]),
                "DFT_TDEP_plus_full_EPC_K_cm-1": float(
                    dft[temperature]["hybrid_cm1"][center]
                ),
                "MLIP_vs_DFT": metrics[temperature],
            }
            for temperature in (300, 450)
        },
        "450K_diagnostics": {
            "old_degauss_0.00285013035_raw_DFT_TDEP_K_cm-1": old450_k,
            "new_degauss_0.0019000869_raw_DFT_TDEP_K_cm-1": new450_raw_k,
            "new_minus_old_smearing_shift_cm-1": new450_raw_k - old450_k,
            "per_seed_raw_K_cm-1": raw_k_markers[450].tolist(),
            "per_seed_raw_K_spread_cm-1": float(np.ptp(raw_k_markers[450])),
            "leave_one_seed_out_hybrid_K_cm-1": [
                float(record["hybrid_cm1"][center]) for record in dft450_variants
            ],
            "leave_one_seed_out_hybrid_K_spread_cm-1": float(
                np.ptp([record["hybrid_cm1"][center] for record in dft450_variants])
            ),
            "source_P4_status": old_acceptance["status"],
            "source_P4_passes_C1": bool(old_acceptance["passes_C1"]),
            "source_P4_failed_metrics": old_acceptance[
                "failed_or_insufficient_metrics"
            ],
        },
        "checks": {
            "operator_geometry_error_A": geometry_error,
            "minimum_branch_overlap": min(
                record["minimum_overlap"]
                for record in [dft300, dft450, *dft300_variants, *dft450_variants]
            ),
            "maximum_FC2_reassembly_error_eV_A2": max(
                record["FC2_reassembly_max_abs_eV_A2"]
                for record in [dft300, dft450, *dft300_variants, *dft450_variants]
            ),
            "all_frequencies_finite": bool(
                all(
                    np.isfinite(record["hybrid_cm1"]).all()
                    for record in [dft300, dft450, *dft300_variants, *dft450_variants]
                )
            ),
        },
        "interpretation": {
            "electronic_smearing_effect_at_450K": (
                "changing degauss from 0.00285013035 to 0.0019000869 Ry moves the "
                "raw DFT-TDEP K frequency by only about -1.13 cm-1"
            ),
            "dominant_failure": (
                "the roughly 40 cm-1 450 K MLIP-DFT gap is not caused by electronic "
                "smearing; the labels are evaluated on an archived failed-C1 MLIP "
                "on-policy ensemble and do not define an equilibrium DFT temperature reference"
            ),
            "allowed_use": (
                "450 K data are a force/model diagnostic and development set, not a "
                "controlled DFT lattice-temperature curve"
            ),
        },
        "outputs": {
            "png": str(png),
            "pdf": str(pdf),
            "csv": str(csv_path),
            "report": str(report_path),
        },
    }
    e48.atomic_json(summary_path, summary)
    near300 = metrics[300]["abs_q_le_0.025_2pi_over_a"]
    near450 = metrics[450]["abs_q_le_0.025_2pi_over_a"]
    report_path.write_text(
        "\n".join(
            [
                "# E51：固定 smearing 的 300/450 K DFT 对比诊断",
                "",
                f"状态：`{summary['status']}`",
                "",
                "## 结果",
                "",
                f"- 300 K K 点：MLIP `{mlip[300][center]:.3f}`，DFT-TDEP+同一 full-EPC `{dft300['hybrid_cm1'][center]:.3f} cm^-1`；近 K 绝对/形状 RMSE `{near300['RMSE_cm-1']:.3f}/{near300['K_referenced_shape_RMSE_cm-1']:.3f} cm^-1`。",
                f"- 450 K K 点：MLIP `{mlip[450][center]:.3f}`，DFT-TDEP+同一 full-EPC `{dft450['hybrid_cm1'][center]:.3f} cm^-1`；近 K 绝对/形状 RMSE `{near450['RMSE_cm-1']:.3f}/{near450['K_referenced_shape_RMSE_cm-1']:.3f} cm^-1`。",
                f"- 450 K 原始 DFT-TDEP 从 `0.00285013035` 改到 `0.0019000869 Ry` 后，K 点只改变 `{new450_raw_k-old450_k:+.3f} cm^-1`。",
                "",
                "## 结论",
                "",
                "450 K 的约 40 cm^-1 差异主要来自采样分布和短程模型，不是 electronic smearing。该批构型来自此前已经 failed-C1 的 MLIP on-policy 轨迹，并非目标 DFT Hamiltonian 的平衡轨迹；三个 seed 的 K 点 spread 也较大。因此 450 K 曲线只能作为模型失配诊断，不能与 300 K DFT-MD 重标注结果组成受控的 DFT 晶格温度趋势。下一步需要调整短程模型/有限温自由能模型，并用少量 DFT 力做 on-policy overlap 检查；在 overlap 通过前不继续堆叠同分布标签。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
