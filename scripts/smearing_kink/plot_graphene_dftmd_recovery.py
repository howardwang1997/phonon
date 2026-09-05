#!/usr/bin/env python3
"""Compare graphene lattice-temperature DFT and MLIP phonon spectra."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import phonopy
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
TD = ROOT / "results" / "td_phonon"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import td_phonon as tdp
import anomaly_locate as al


CM_PER_THZ = 33.356


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)


def load_temperature(path: Path, temperature: int):
    data = np.load(path, allow_pickle=False)
    prefix = f"T{temperature}"
    return (
        data,
        np.asarray(data[f"{prefix}_dist"], float),
        np.asarray(data[f"{prefix}_freq"], float) * CM_PER_THZ,
    )


def interpolate(reference_distance, reference_frequency, target_distance):
    return np.stack(
        [
            np.interp(target_distance, reference_distance, reference_frequency[:, branch])
            for branch in range(reference_frequency.shape[1])
        ],
        axis=1,
    )


def mae(reference_distance, reference_frequency, candidate_distance, candidate_frequency):
    reference_on_candidate = interpolate(
        reference_distance, reference_frequency, candidate_distance
    )
    return float(np.mean(np.abs(candidate_frequency - reference_on_candidate)))


def label_index(distance, label_positions, labels, wanted):
    normalized = [str(label).replace("$", "").replace("\\", "") for label in labels]
    index = normalized.index(wanted)
    return int(np.argmin(np.abs(distance - label_positions[index])))


def convergence_curve(path: Path):
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    prefix = [
        record for record in payload["records"] if record["subset"].startswith("prefix_")
    ]
    prefix.sort(key=lambda record: record["n_snapshots"])
    return payload, prefix


def k_kink_on_grid(distance, frequency_cm, label_positions, labels) -> float:
    records = {
        record["label"]: record
        for record in al.high_sym_kinks(
            distance, frequency_cm / CM_PER_THZ, label_positions, labels
        )
    }
    return float(records["K"]["kink_strength"])


def main() -> int:
    dft300_data, dft300_distance, dft300_frequency = load_temperature(
        TD / "graphene_dft_tdep_300K_recovery.npz", 300
    )
    plain_data, plain_distance, plain_frequency = load_temperature(
        TD / "td_graphene_ft.npz", 300
    )
    lr_data, lr_distance, lr_frequency = load_temperature(
        TD / "td_graphene_ft_friedel.npz", 300
    )

    harmonic_path = ROOT / "results" / "vq_kink6" / "graphene_sc6_dg0.005_phonopy.yaml"
    harmonic = phonopy.load(harmonic_path, is_compact_fc=False)
    harmonic_distance, harmonic_frequency, harmonic_ticks, harmonic_labels = (
        tdp.band_from_phonopy(
            harmonic, np.asarray(harmonic.force_constants), npoints=60
        )
    )
    harmonic_frequency = harmonic_frequency * CM_PER_THZ

    spectra = [
        ("0 K harmonic DFT", harmonic_distance, harmonic_frequency, "#777777", "--"),
        ("300 K DFT-MD/TDEP", dft300_distance, dft300_frequency, "#222222", "-"),
    ]
    dft600_path = TD / "graphene_dft_tdep_600K_recovery.npz"
    dft600 = None
    if dft600_path.is_file():
        dft600_data, dft600_distance, dft600_frequency = load_temperature(
            dft600_path, 600
        )
        dft600 = (dft600_data, dft600_distance, dft600_frequency)
        spectra.append(
            ("600 K DFT-MD/TDEP", dft600_distance, dft600_frequency, "#0072B2", "-")
        )

    zero_to_300 = mae(
        dft300_distance,
        dft300_frequency,
        harmonic_distance,
        harmonic_frequency,
    )
    plain_mae = mae(
        dft300_distance, dft300_frequency, plain_distance, plain_frequency
    )
    lr_mae = mae(dft300_distance, dft300_frequency, lr_distance, lr_frequency)

    ticks = np.asarray(dft300_data["label_positions"], float)
    labels = [str(label) for label in dft300_data["labels"]]
    gamma_index = label_index(dft300_distance, ticks, labels, "Gamma")
    k_index = label_index(dft300_distance, ticks, labels, "K")
    plain_on_dft_grid = interpolate(plain_distance, plain_frequency, dft300_distance)
    lr_on_dft_grid = interpolate(lr_distance, lr_frequency, dft300_distance)
    dft300_k_kink = k_kink_on_grid(
        dft300_distance, dft300_frequency, ticks, labels
    )
    plain_k_kink = k_kink_on_grid(
        dft300_distance, plain_on_dft_grid, ticks, labels
    )
    lr_k_kink = k_kink_on_grid(dft300_distance, lr_on_dft_grid, ticks, labels)
    dft300_to_600 = None
    dft600_on_dft_grid = None
    dft600_k_kink = None
    if dft600 is not None:
        _, dft600_distance, dft600_frequency = dft600
        dft300_to_600 = mae(
            dft300_distance,
            dft300_frequency,
            dft600_distance,
            dft600_frequency,
        )
        dft600_on_dft_grid = interpolate(
            dft600_distance, dft600_frequency, dft300_distance
        )
        dft600_k_kink = k_kink_on_grid(
            dft300_distance, dft600_on_dft_grid, ticks, labels
        )

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.1))

    ax = axes[0]
    for name, distance, frequency, color, linestyle in spectra:
        ax.plot(
            distance,
            frequency,
            color=color,
            ls=linestyle,
            lw=1.35 if name.startswith("0 K") else 1.55,
            alpha=0.9,
        )
    ax.set_title(
        "Lattice-temperature comparison"
        + (": 0/300/600 K" if dft600 is not None else ": 0/300 K"),
        fontsize=10,
        pad=32,
    )
    ax.set_ylabel(r"frequency (cm$^{-1}$)")
    lattice_annotation = (
        rf"0→300 K full-band MAE = {zero_to_300:.1f} cm$^{{-1}}$"
        "\n"
        rf"300 K: $\Gamma_{{max}}$={dft300_frequency[gamma_index, -1]:.0f}, "
        rf"K$_{{max}}$={dft300_frequency[k_index, -1]:.0f} cm$^{{-1}}$"
    )
    if dft600_on_dft_grid is not None:
        lattice_annotation = (
            rf"0→300 / 300→600 K full-band MAE = "
            rf"{zero_to_300:.1f} / {dft300_to_600:.1f} cm$^{{-1}}$"
            "\n"
            rf"$\Gamma_{{max}}$: 300 K {dft300_frequency[gamma_index, -1]:.0f}; "
            rf"600 K {dft600_on_dft_grid[gamma_index, -1]:.0f} cm$^{{-1}}$"
            "\n"
            rf"K$_{{max}}$: 300 K {dft300_frequency[k_index, -1]:.0f}; "
            rf"600 K {dft600_on_dft_grid[k_index, -1]:.0f} cm$^{{-1}}$"
        )
    ax.legend(
        handles=[
            Line2D([0], [0], color=color, ls=linestyle, lw=1.5, label=name)
            for name, _, _, color, linestyle in spectra
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=7.7,
        ncol=2,
    )
    style(ax)

    ax = axes[1]
    ax.plot(
        dft300_distance,
        dft300_frequency,
        color="#222222",
        lw=1.65,
        alpha=0.9,
    )
    ax.plot(
        plain_distance,
        plain_frequency,
        color="#D55E00",
        lw=1.05,
        ls="--",
        alpha=0.82,
    )
    ax.plot(
        lr_distance,
        lr_frequency,
        color="#0072B2",
        lw=1.05,
        ls=":",
        alpha=0.88,
    )
    ax.set_title("300 K model comparison (different backbones)", fontsize=10, pad=32)
    model_annotation = (
        rf"plain fine-tuned MLIP MAE = {plain_mae:.1f} cm$^{{-1}}$"
        "\n"
        rf"v11 + long-range MAE = {lr_mae:.1f} cm$^{{-1}}$"
        "\n"
        rf"K kink $|\Delta v|$: DFT {dft300_k_kink:.1f}, v11+LR {lr_k_kink:.1f}"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#222222", lw=1.65, label="300 K DFT-MD/TDEP"),
            Line2D(
                [0], [0], color="#D55E00", lw=1.05, ls="--",
                label="fine-tuned MLIP (plain)",
            ),
            Line2D(
                [0], [0], color="#0072B2", lw=1.05, ls=":",
                label="v11 backbone + long-range",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=7.6,
        ncol=2,
    )
    style(ax)

    for ax in axes[:2]:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        for tick in ticks[1:-1]:
            ax.axvline(tick, color="#d8d8d8", lw=0.65, zorder=0)
        ax.axhline(0.0, color="#999999", lw=0.65)
        ax.set_xlim(float(dft300_distance[0]), float(dft300_distance[-1]))
        ax.set_ylim(-25, 1650)

    ax = axes[2]
    convergence_specs = [
        (
            300,
            TD / "graphene_dft_tdep_300K_recovery_convergence.json",
            "#222222",
            "o-",
        )
    ]
    if dft600 is not None:
        convergence_specs.append(
            (
                600,
                TD / "graphene_dft_tdep_600K_recovery_convergence.json",
                "#0072B2",
                "s--",
            )
        )
    convergence_metrics = {}
    for temperature, path, color, marker in convergence_specs:
        loaded = convergence_curve(path)
        if loaded is None:
            continue
        payload, prefix = loaded
        count = np.asarray([record["n_snapshots"] for record in prefix])
        band_mae = np.asarray(
            [record["band_MAE_vs_full_cm-1"] for record in prefix]
        )
        ax.plot(
            count,
            band_mae,
            marker,
            color=color,
            lw=1.35,
            ms=4.2,
            label=f"{temperature} K",
        )
        records = {record["subset"]: record for record in payload["records"]}
        convergence_metrics[str(temperature)] = {
            "prefix_50_to_60_band_MAE_cm-1": float(
                records.get("prefix_50", {}).get("band_MAE_vs_full_cm-1", np.nan)
            ),
            "first_last_30_K_highest_difference_cm-1": float(
                abs(
                    records.get("first_30", {}).get("K_highest_cm-1", np.nan)
                    - records.get("last_30", {}).get("K_highest_cm-1", np.nan)
                )
            ),
        }
    ax.set_xlabel("number of DFT-MD snapshots", labelpad=3)
    ax.set_ylabel(r"full-band MAE vs 60 snapshots (cm$^{-1}$)")
    ax.set_title("Snapshot convergence", fontsize=10, pad=32)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    annotation = []
    for temperature, values in convergence_metrics.items():
        annotation.append(
            f"{temperature} K: 50→60 MAE "
            f"{values['prefix_50_to_60_band_MAE_cm-1']:.2f}; "
            f"K first/last Δ {values['first_last_30_K_highest_difference_cm-1']:.1f} cm⁻¹"
        )
    convergence_note = "\n".join(annotation)
    style(ax)

    metrics = {
        "harmonic_reference": str(harmonic_path.relative_to(ROOT)),
        "electronic_smearing_reference": "cold, degauss=0.005 Ry",
        "MLIP_plain_model": str(np.asarray(plain_data["model"]).item()),
        "MLIP_long_range_model": str(np.asarray(lr_data["model"]).item()),
        "MLIP_comparison_uses_same_backbone": False,
        "dft_lattice_temperatures_K": [300] + ([600] if dft600 is not None else []),
        "harmonic_0K_vs_DFTMD_300K_full_band_MAE_cm-1": zero_to_300,
        "MLIP_300K_vs_DFTMD_300K_full_band_MAE_cm-1": plain_mae,
        "MLIP_long_range_300K_vs_DFTMD_300K_full_band_MAE_cm-1": lr_mae,
        "DFTMD_300K_Gamma_highest_cm-1": float(dft300_frequency[gamma_index, -1]),
        "DFTMD_300K_K_highest_cm-1": float(dft300_frequency[k_index, -1]),
        "K_kink_common_grid_unit": "THz per q-path-distance",
        "DFTMD_300K_K_kink": dft300_k_kink,
        "MLIP_300K_K_kink": plain_k_kink,
        "MLIP_long_range_300K_K_kink": lr_k_kink,
        "convergence": convergence_metrics,
    }
    if dft600 is not None:
        metrics["DFTMD_300K_vs_600K_full_band_MAE_cm-1"] = dft300_to_600
        metrics["DFTMD_600K_Gamma_highest_cm-1"] = float(
            dft600_on_dft_grid[gamma_index, -1]
        )
        metrics["DFTMD_600K_K_highest_cm-1"] = float(
            dft600_on_dft_grid[k_index, -1]
        )
        metrics["DFTMD_600K_K_kink"] = dft600_k_kink
    (TD / "graphene_lattice_temperature_summary.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    fig.suptitle(
        "Graphene lattice-temperature phonons: DFT reference and MLIP comparison",
        fontsize=11.5,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.22, 1, 0.86), w_pad=1.5)
    footnotes = (lattice_annotation, model_annotation, convergence_note)
    footnote_sizes = (7.8, 7.8, 7.4)
    for axis, note, size in zip(axes, footnotes, footnote_sizes):
        if not note:
            continue
        axis_box = axis.get_position()
        fig.text(
            axis_box.x0,
            0.035,
            note,
            fontsize=size,
            va="bottom",
        )
    for stem in (
        "graphene_lattice_temperature_summary",
        "graphene_dftmd_300K_recovery_summary",
    ):
        for suffix in ("png", "pdf"):
            fig.savefig(TD / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print("wrote graphene_lattice_temperature_summary.{png,pdf,json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
