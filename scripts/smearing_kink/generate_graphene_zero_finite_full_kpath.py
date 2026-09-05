#!/usr/bin/env python3
"""Generate a complete graphene K-neighbourhood A' comparison.

The figure combines the existing zero-smearing K->Gamma and K->M curve with
new finite-smearing evaluations on exactly the same two directions.  Finite
curves use the accepted L0 short-range force constants and the full stored
Wannier EPC vertex.  The K->M side is evaluated directly, never mirrored from
K->Gamma.  No new DFT calculation or empirical q-shape fit is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_graphene_k_cusp_two_methods as a0  # noqa: E402
import fit_graphene_joint_zero_finite_rank1 as joint  # noqa: E402
from ablate_graphene_constant_vertex_bandsum import (  # noqa: E402
    RYDBERG_EV,
    fermi,
    hamiltonian_grid,
)
from diagnose_graphene_epw_wannier import read_hr  # noqa: E402
from evaluate_graphene_epc_zero_adaptive import (  # noqa: E402
    dense_realspace_array,
    fft_from_realspace,
    interpolate_rectangle,
    mode_projected_coefficients,
)
from evaluate_graphene_epc_zero_transfer import qpoint  # noqa: E402
from extract_graphene_epw_dynamical_matrices import (  # noqa: E402
    RYDBERG_CM1,
    load_crystal,
    load_rdw,
    zone_centered_wigner,
)
from replay_graphene_epc_bandsum import (  # noqa: E402
    dynamical_matrix_and_mode,
    load_epmatwp,
    signed_frequency,
)


BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
TEMPERATURES_K = (300, 450, 600)
COLORS = ("#111111", "#0072B2", "#009E73", "#D55E00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-nk", type=int, default=144)
    parser.add_argument("--patch-n", type=int, default=180)
    parser.add_argument("--patch-halfwidth", type=float, default=0.04)
    parser.add_argument("--eta-eV", type=float, default=0.005)
    parser.add_argument("--maximum-distance", type=float, default=0.03)
    parser.add_argument("--points-per-direction", type=int, default=61)
    parser.add_argument("--fermi-eV", type=float, default=-1.7187)
    parser.add_argument("--reference-degauss-Ry", type=float, default=0.020000)
    parser.add_argument(
        "--zero-curve",
        type=Path,
        default=BASE / "E15_epc_zero_dense_curve/zero_dense_curve.csv",
    )
    parser.add_argument(
        "--finite-replay",
        type=Path,
        default=(
            BASE
            / "E9_epc_bandsum_replay_nk720/epc_bandsum_replay_arrays.npz"
        ),
    )
    parser.add_argument(
        "--hr",
        type=Path,
        default=BASE
        / "E0_epw_matched/wannier_diagnostics/matched_k18q9_ex1_pifroz_hr.dat",
    )
    parser.add_argument(
        "--epwdata",
        type=Path,
        default=BASE / "E0_epw_matched/k18_q9_ex1_pifroz/restart_small/epwdata.fmt",
    )
    parser.add_argument(
        "--crystal",
        type=Path,
        default=BASE / "E0_epw_matched/k18_q9_ex1_pifroz/restart_small/crystal.fmt",
    )
    parser.add_argument(
        "--epmatwp",
        type=Path,
        default=BASE / "E9_epc_bandsum_assets/graphene.epmatwp",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "E30_zero_finite_full_kpath",
    )
    return parser.parse_args()


def response_integrands(
    initial_energy_eV: np.ndarray,
    initial_vectors: np.ndarray,
    final_energy_eV: np.ndarray,
    final_vectors: np.ndarray,
    vertex_wannier: np.ndarray,
    smearings_Ry: np.ndarray,
    reference_Ry: float,
    fermi_eV: float,
    eta_eV: float,
) -> np.ndarray:
    """Return band-summed integrands with shape (n_smearing, ...k_grid)."""
    vertex_band = np.einsum(
        "...am,...ab,...bn->...mn",
        final_vectors.conj(),
        vertex_wannier,
        initial_vectors,
        optimize=True,
    )
    g_squared = np.abs(vertex_band) ** 2
    initial = initial_energy_eV[..., None, :]
    final = final_energy_eV[..., :, None]
    delta_Ry = (final - initial) / RYDBERG_EV
    denominator = np.real(1.0 / (delta_Ry + 1j * eta_eV / RYDBERG_EV))
    reference_width_eV = reference_Ry * RYDBERG_EV
    reference_fact = fermi(final, fermi_eV, reference_width_eV) - fermi(
        initial, fermi_eV, reference_width_eV
    )
    values = []
    for smearing_Ry in smearings_Ry:
        width_eV = float(smearing_Ry) * RYDBERG_EV
        target_fact = fermi(final, fermi_eV, width_eV) - fermi(
            initial, fermi_eV, width_eV
        )
        values.append(
            np.sum(
                g_squared * (target_fact - reference_fact) * denominator,
                axis=(-2, -1),
            )
        )
    return np.asarray(values, float)


def load_zero_curve(
    path: Path, expected_qpoints: int
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_qpoints:
        raise RuntimeError(
            f"expected {expected_qpoints} zero-smearing q points, "
            f"found {len(rows)}"
        )
    return rows


def make_records(
    maximum_distance: float, points_per_direction: int
) -> tuple[list[tuple[str, float]], np.ndarray]:
    distances = np.linspace(0.0, maximum_distance, points_per_direction)
    records: list[tuple[str, float]] = [("K", 0.0)]
    records.extend(("KG", float(value)) for value in distances[1:])
    records.extend(("KM", float(value)) for value in distances[1:])
    qpoints = np.asarray(
        [qpoint(direction, distance) for direction, distance in records], float
    )
    return records, qpoints


def load_finite_anchors(path: Path) -> dict:
    with np.load(path) as arrays:
        t_gk = np.asarray(arrays["t_GK"], float)
        smearings = np.asarray(arrays["degauss_Ry"], float)
        target_correction = np.asarray(arrays["target_correction_cm2"], float)
        replay_correction = np.asarray(arrays["replay_correction_cm2"], float)
        background = np.asarray(arrays["background_cm1"], float)
    center = int(np.argmin(np.abs(t_gk - 1.0)))
    target_frequency = signed_frequency(background, target_correction)
    anchored_replay_correction = (
        target_correction[:, [center]]
        + replay_correction
        - replay_correction[:, [center]]
    )
    replay_frequency = signed_frequency(background, anchored_replay_correction)
    return {
        "t_gk": t_gk,
        "smearings_Ry": smearings,
        "K_frequency_cm1": target_frequency[:, center],
        "background_cm1": background,
        "replay_frequency_cm1": replay_frequency,
    }


def finite_short_backgrounds(qpoints: np.ndarray) -> np.ndarray:
    backgrounds = []
    for temperature in TEMPERATURES_K:
        operator_path = a0.operator_path(temperature)
        with np.load(operator_path, allow_pickle=False) as payload:
            operator = {key: np.asarray(payload[key]) for key in payload.files}
        phonon, _ = a0.make_phonopy(operator)
        finite_path, force_constant_key = a0.finite_lattice_path(temperature, "L0")
        with np.load(finite_path, allow_pickle=False) as payload:
            total_force_constants = np.asarray(payload[force_constant_key], float)
        short_force_constants = total_force_constants - np.asarray(
            operator["delta_fc_full"], float
        )
        backgrounds.append(
            joint.tracked_static_background(phonon, short_force_constants, qpoints)
        )
    return np.asarray(backgrounds, float)


def evaluate_finite_response(
    args: argparse.Namespace,
    qpoints: np.ndarray,
    smearings_Ry: np.ndarray,
) -> np.ndarray:
    crystal = load_crystal(args.crystal, natoms=2)
    rdw, dimensions = load_rdw(args.epwdata)
    vectors_k, degeneracies_k, _ = zone_centered_wigner(
        crystal["at"], (18, 18, 1)
    )
    vectors_q, degeneracies_q, _ = zone_centered_wigner(
        crystal["at"], (9, 9, 1)
    )
    translations, hamiltonian_matrices = read_hr(args.hr)
    if not np.array_equal(translations, vectors_k):
        raise RuntimeError("Hamiltonian and EPC electronic WS orders differ")
    epmatwp = load_epmatwp(
        args.epmatwp,
        dimensions["nbndsub"],
        dimensions["nrr_k"],
        dimensions["nmodes"],
        dimensions["nrr_g"],
    )

    coarse_initial_h = hamiltonian_grid(
        translations, hamiltonian_matrices, args.coarse_nk, np.zeros(3)
    )
    coarse_initial_energy, coarse_initial_vectors = np.linalg.eigh(
        coarse_initial_h
    )
    coarse_coordinates = np.arange(args.coarse_nk, dtype=float) / args.coarse_nk
    coarse_k1, coarse_k2 = np.meshgrid(
        coarse_coordinates, coarse_coordinates, indexing="ij"
    )
    k_value = 1.0 / 3.0
    coarse_patch_mask = (
        (np.abs(coarse_k1 - k_value) < args.patch_halfwidth)
        & (np.abs(coarse_k2 - k_value) < args.patch_halfwidth)
    )

    patch_step = 2.0 * args.patch_halfwidth / args.patch_n
    patch_lower = (
        k_value
        - args.patch_halfwidth
        + np.arange(args.patch_n, dtype=float) * patch_step
    )
    quadrature_grids = [
        (
            patch_lower + (2.0 / 3.0) * patch_step,
            patch_lower + (1.0 / 3.0) * patch_step,
            0.5,
        ),
        (
            patch_lower + (1.0 / 3.0) * patch_step,
            patch_lower + (2.0 / 3.0) * patch_step,
            0.5,
        ),
    ]
    r1_h, r2_h, dense_h = dense_realspace_array(
        translations, np.moveaxis(hamiltonian_matrices, 0, -1)
    )
    patch_initial_states = []
    for coordinates1, coordinates2, weight in quadrature_grids:
        initial_h = interpolate_rectangle(
            r1_h, r2_h, dense_h, coordinates1, coordinates2
        )
        initial_h = 0.5 * (initial_h + np.swapaxes(initial_h.conj(), -1, -2))
        initial_energy, initial_vectors = np.linalg.eigh(initial_h)
        patch_initial_states.append((initial_energy, initial_vectors, weight))

    response = np.empty((len(smearings_Ry), len(qpoints)), float)
    patch_area = (2.0 * args.patch_halfwidth) ** 2
    for q_index, qpoint_value in enumerate(qpoints):
        _, _, physical_mode = dynamical_matrix_and_mode(
            rdw,
            qpoint_value,
            vectors_q,
            degeneracies_q,
            crystal["masses_au"],
            crystal["atom_types"],
        )
        g_coefficients = mode_projected_coefficients(
            epmatwp,
            qpoint_value,
            vectors_q,
            degeneracies_q,
            physical_mode,
            degeneracies_k,
        )
        coarse_vertex = fft_from_realspace(
            vectors_k, g_coefficients, args.coarse_nk
        )
        coarse_final_h = hamiltonian_grid(
            translations, hamiltonian_matrices, args.coarse_nk, qpoint_value
        )
        coarse_final_energy, coarse_final_vectors = np.linalg.eigh(coarse_final_h)
        coarse_integrand = response_integrands(
            coarse_initial_energy,
            coarse_initial_vectors,
            coarse_final_energy,
            coarse_final_vectors,
            coarse_vertex,
            smearings_Ry,
            args.reference_degauss_Ry,
            args.fermi_eV,
            args.eta_eV,
        )
        coarse_response = 2.0 * np.mean(coarse_integrand, axis=(1, 2))
        coarse_patch_response = (
            2.0
            * np.sum(coarse_integrand[:, coarse_patch_mask], axis=1)
            / args.coarse_nk**2
        )

        h_shift_phase = np.exp(2j * np.pi * (translations @ qpoint_value))
        shifted_h_coefficients = np.moveaxis(
            hamiltonian_matrices * h_shift_phase[:, None, None], 0, -1
        )
        r1_g, r2_g, dense_g = dense_realspace_array(vectors_k, g_coefficients)
        r1_q, r2_q, dense_hq = dense_realspace_array(
            translations, shifted_h_coefficients
        )
        if not np.array_equal(r1_g, r1_q) or not np.array_equal(r2_g, r2_q):
            raise RuntimeError("local Hamiltonian and EPC grids differ")

        patch_integrand_mean = np.zeros(len(smearings_Ry), float)
        for grid_index, (coordinates1, coordinates2, weight) in enumerate(
            quadrature_grids
        ):
            final_h = interpolate_rectangle(
                r1_q, r2_q, dense_hq, coordinates1, coordinates2
            )
            final_h = 0.5 * (final_h + np.swapaxes(final_h.conj(), -1, -2))
            final_energy, final_vectors = np.linalg.eigh(final_h)
            vertex = interpolate_rectangle(
                r1_g, r2_g, dense_g, coordinates1, coordinates2
            )
            initial_energy, initial_vectors, initial_weight = patch_initial_states[
                grid_index
            ]
            if initial_weight != weight:
                raise RuntimeError("quadrature weights are inconsistent")
            integrand = response_integrands(
                initial_energy,
                initial_vectors,
                final_energy,
                final_vectors,
                vertex,
                smearings_Ry,
                args.reference_degauss_Ry,
                args.fermi_eV,
                args.eta_eV,
            )
            patch_integrand_mean += weight * np.mean(integrand, axis=(1, 2))
        fine_patch_response = 2.0 * patch_area * patch_integrand_mean
        response[:, q_index] = (
            coarse_response - coarse_patch_response + fine_patch_response
        )
    return response


def direction_indices(records: list[tuple[str, float]], direction: str) -> list[int]:
    return [0] + [
        index for index, (label, _) in enumerate(records) if label == direction
    ]


def signed_axis(records: list[tuple[str, float]]) -> np.ndarray:
    return np.asarray(
        [
            0.0
            if direction == "K"
            else (-distance if direction == "KG" else distance)
            for direction, distance in records
        ],
        float,
    )


def validate_against_nk720(
    records: list[tuple[str, float]],
    finite_frequency: np.ndarray,
    anchors: dict,
) -> dict:
    kg_indices = direction_indices(records, "KG")
    model_distance = np.asarray([records[index][1] for index in kg_indices], float)
    t_gk = anchors["t_gk"]
    mask = (t_gk <= 1.0 + 1.0e-12) & (1.0 - t_gk <= model_distance.max() + 1e-12)
    validation_distance = 1.0 - t_gk[mask]
    order = np.argsort(validation_distance)
    validation_distance = validation_distance[order]
    result = {}
    for temperature_index, temperature in enumerate(TEMPERATURES_K):
        interpolator = PchipInterpolator(
            model_distance, finite_frequency[temperature_index, kg_indices]
        )
        prediction = np.asarray(interpolator(validation_distance), float)
        reference = anchors["replay_frequency_cm1"][temperature_index, mask][order]
        error = prediction - reference
        result[str(temperature)] = {
            "number_of_points": len(validation_distance),
            "RMSE_cm-1": float(np.sqrt(np.mean(error**2))),
            "maximum_absolute_error_cm-1": float(np.max(np.abs(error))),
        }
    return result


def curve_metrics(
    records: list[tuple[str, float]],
    smearings: np.ndarray,
    frequencies: np.ndarray,
) -> list[dict]:
    output = []
    for series_index, smearing in enumerate(smearings):
        row = {
            "smearing_degauss_Ry": float(smearing),
            "lattice_temperature_K": (
                None if series_index == 0 else TEMPERATURES_K[series_index - 1]
            ),
            "K_frequency_cm-1": float(frequencies[series_index, 0]),
        }
        for direction in ("KG", "KM"):
            index = next(
                index
                for index, (label, distance) in enumerate(records)
                if label == direction and np.isclose(distance, 0.003, atol=1e-14)
            )
            row[f"{direction}_d003_rise_cm-1"] = float(
                frequencies[series_index, index] - frequencies[series_index, 0]
            )
        output.append(row)
    return output


def make_figure(
    path_png: Path,
    path_pdf: Path,
    records: list[tuple[str, float]],
    smearings: np.ndarray,
    frequencies: np.ndarray,
    anchors: dict,
) -> None:
    x = signed_axis(records)
    labels = ["0 Ry"] + [
        f"{smearing:.7f} Ry; lattice {temperature} K"
        for smearing, temperature in zip(smearings[1:], TEMPERATURES_K)
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14.8, 8.8))
    absolute_axis, centered_axis, zoom_axis, depth_axis = axes.flat
    for series_index, (label, color) in enumerate(zip(labels, COLORS)):
        frequency = frequencies[series_index]
        order = np.argsort(x)
        linewidth = 2.7 if series_index == 0 else 2.1
        absolute_axis.plot(
            x[order], frequency[order], color=color, lw=linewidth, label=label
        )
        centered = frequency - frequency[0]
        centered_axis.plot(x[order], centered[order], color=color, lw=linewidth)
        zoom_axis.plot(x[order], centered[order], color=color, lw=linewidth)
        if series_index > 0:
            for direction in ("KG", "KM"):
                indices = direction_indices(records, direction)[::10]
                if indices[-1] != direction_indices(records, direction)[-1]:
                    indices.append(direction_indices(records, direction)[-1])
                absolute_axis.plot(
                    x[indices],
                    frequency[indices],
                    "o",
                    color=color,
                    ms=3.0,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    zorder=3,
                )

    t_gk = anchors["t_gk"]
    validation_mask = (t_gk <= 1.0 + 1.0e-12) & (1.0 - t_gk <= 0.0300001)
    validation_x = t_gk[validation_mask] - 1.0
    for finite_index, color in enumerate(COLORS[1:]):
        absolute_axis.plot(
            validation_x,
            anchors["replay_frequency_cm1"][finite_index, validation_mask],
            "o",
            ms=4.0,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=0.8,
            alpha=0.8,
        )

    metrics = curve_metrics(records, smearings, frequencies)
    for direction, marker, linestyle in (("KG", "o", "-"), ("KM", "s", "--")):
        depth_axis.plot(
            [row["smearing_degauss_Ry"] for row in metrics],
            [row[f"{direction}_d003_rise_cm-1"] for row in metrics],
            marker=marker,
            ls=linestyle,
            lw=1.8,
            ms=5.0,
            color="#4A4A4A" if direction == "KG" else "#8A4F9E",
            label=f"K→{'Γ' if direction == 'KG' else 'M'}",
        )

    absolute_axis.set_title("Absolute frequency: Γ–K–M")
    centered_axis.set_title("K-referenced cusp: full window")
    zoom_axis.set_title("K-referenced cusp: inner window")
    depth_axis.set_title(r"Cusp rise at $|q-K|=0.003$")
    absolute_axis.set_ylabel(r"A$'$ frequency (cm$^{-1}$)")
    centered_axis.set_ylabel(r"$\omega(q)-\omega(K)$ (cm$^{-1}$)")
    zoom_axis.set_ylabel(r"$\omega(q)-\omega(K)$ (cm$^{-1}$)")
    depth_axis.set_ylabel(r"$\omega(q)-\omega(K)$ (cm$^{-1}$)")
    for axis in (absolute_axis, centered_axis, zoom_axis):
        axis.axvline(0.0, color="#AAAAAA", lw=0.9, zorder=0)
        axis.set_xlabel("signed distance from K (K→Γ < 0; K→M > 0)")
        axis.grid(alpha=0.18)
    absolute_axis.set_xlim(-0.03, 0.03)
    centered_axis.set_xlim(-0.03, 0.03)
    zoom_axis.set_xlim(-0.008, 0.008)
    inner_mask = np.abs(x) <= 0.008 + 1.0e-12
    inner_maximum = float(
        np.max(frequencies[:, inner_mask] - frequencies[:, [0]])
    )
    zoom_axis.set_ylim(-0.08, 1.08 * inner_maximum)
    depth_axis.set_xlabel("smearing/degauss (Ry)")
    depth_axis.grid(alpha=0.18)
    depth_axis.legend(loc="best", frameon=False)
    figure.suptitle(
        "Graphene K-neighbourhood A′: MLIP short range + full EPC long range",
        fontsize=16,
        y=0.985,
    )
    handles, legend_labels = absolute_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.79, 0.68),
        frameon=False,
        title="smearing/degauss (Ry)",
    )
    figure.text(
        0.79,
        0.47,
        "Finite curves: L0 short-range backgrounds\n"
        "+ direct full-EPC q-space response.\n"
        "Filled dots: calculated q samples.\n"
        "Open dots: existing nk=720 K→Γ replay.\n"
        "K→M is calculated directly, not mirrored.",
        fontsize=9.2,
        color="#555555",
        ha="left",
        va="top",
    )
    figure.text(
        0.79,
        0.27,
        "Absolute finite-smearing shifts also include\n"
        "the paired 300/450/600 K lattice backgrounds;\n"
        "use the K-referenced panels for cusp shape.",
        fontsize=9.2,
        color="#555555",
        ha="left",
        va="top",
    )
    figure.subplots_adjust(
        left=0.07, right=0.77, bottom=0.09, top=0.90, wspace=0.25, hspace=0.33
    )
    figure.savefig(path_png, dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(path_pdf, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.coarse_nk % 3 != 0:
        raise ValueError("coarse_nk must be divisible by 3")
    if args.points_per_direction < 7:
        raise ValueError("points_per_direction is too small")
    started = time.perf_counter()
    expected_qpoints = 2 * args.points_per_direction - 1
    zero_rows = load_zero_curve(args.zero_curve, expected_qpoints)
    records, qpoints = make_records(
        args.maximum_distance, args.points_per_direction
    )
    zero_qpoints = np.asarray(
        [
            [float(row["q1"]), float(row["q2"]), 0.0]
            for row in zero_rows
        ],
        float,
    )
    if not np.allclose(qpoints, zero_qpoints, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("zero curve and requested full K path do not match")
    zero_frequency = np.asarray(
        [float(row["MLIP_plus_full_EPC_cm-1"]) for row in zero_rows], float
    )
    zero_background = np.asarray(
        [float(row["short_range_MLIP_cm-1"]) for row in zero_rows], float
    )
    zero_response = np.asarray(
        [float(row["full_EPC_response_Ry2"]) for row in zero_rows], float
    )
    zero_correction = np.asarray(
        [float(row["long_range_correction_cm-2"]) for row in zero_rows], float
    )

    anchors = load_finite_anchors(args.finite_replay)
    smearings_finite = anchors["smearings_Ry"]
    if len(smearings_finite) != len(TEMPERATURES_K):
        raise RuntimeError("unexpected finite-smearing series count")
    finite_response_Ry2 = evaluate_finite_response(
        args, qpoints, smearings_finite
    )
    finite_background = finite_short_backgrounds(qpoints)
    finite_response_cm2 = finite_response_Ry2 * RYDBERG_CM1**2
    finite_correction = np.empty_like(finite_response_cm2)
    finite_frequency = np.empty_like(finite_response_cm2)
    for index in range(len(smearings_finite)):
        K_correction = (
            anchors["K_frequency_cm1"][index] ** 2
            - finite_background[index, 0] ** 2
        )
        finite_correction[index] = (
            K_correction
            + finite_response_cm2[index]
            - finite_response_cm2[index, 0]
        )
        finite_frequency[index] = signed_frequency(
            finite_background[index], finite_correction[index]
        )

    smearings = np.concatenate(([0.0], smearings_finite))
    frequencies = np.vstack((zero_frequency, finite_frequency))
    backgrounds = np.vstack((zero_background, finite_background))
    responses = np.vstack((zero_response, finite_response_Ry2))
    corrections = np.vstack((zero_correction, finite_correction))
    validation = validate_against_nk720(records, finite_frequency, anchors)
    metrics = curve_metrics(records, smearings, frequencies)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "zero_finite_full_kpath_Aprime.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "smearing_degauss_Ry",
            "lattice_temperature_K",
            "direction",
            "distance",
            "signed_distance",
            "q1",
            "q2",
            "short_range_background_cm-1",
            "full_EPC_response_Ry2",
            "long_range_correction_cm-2",
            "MLIP_plus_full_EPC_cm-1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        signed = signed_axis(records)
        for series_index, smearing in enumerate(smearings):
            temperature = None if series_index == 0 else TEMPERATURES_K[series_index - 1]
            for q_index, ((direction, distance), qpoint_value) in enumerate(
                zip(records, qpoints)
            ):
                writer.writerow(
                    {
                        "smearing_degauss_Ry": float(smearing),
                        "lattice_temperature_K": "" if temperature is None else temperature,
                        "direction": direction,
                        "distance": distance,
                        "signed_distance": signed[q_index],
                        "q1": qpoint_value[0],
                        "q2": qpoint_value[1],
                        "short_range_background_cm-1": backgrounds[
                            series_index, q_index
                        ],
                        "full_EPC_response_Ry2": responses[series_index, q_index],
                        "long_range_correction_cm-2": corrections[
                            series_index, q_index
                        ],
                        "MLIP_plus_full_EPC_cm-1": frequencies[
                            series_index, q_index
                        ],
                    }
                )

    png_path = output / "zero_finite_full_kpath_Aprime_comparison.png"
    pdf_path = output / "zero_finite_full_kpath_Aprime_comparison.pdf"
    make_figure(png_path, pdf_path, records, smearings, frequencies, anchors)
    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete_zero_finite_full_EPC_K_neighbourhood_generated",
        "scope": (
            "graphene A-prime on K-to-Gamma and K-to-M; short-range MLIP/L0 "
            "background plus direct full-Wannier-EPC q-space response; no new DFT"
        ),
        "integration": {
            "coarse_nk": args.coarse_nk,
            "patch_quadrature": "triangle_centroid",
            "patch_n": args.patch_n,
            "patch_halfwidth_fractional": args.patch_halfwidth,
            "finite_eta_eV": args.eta_eV,
            "number_of_qpoints": len(qpoints),
            "number_of_finite_smearings": len(smearings_finite),
            "wall_time_seconds": elapsed,
        },
        "series_metrics": metrics,
        "K_to_Gamma_validation_against_existing_nk720_replay": validation,
        "checks": {
            "K_to_M_evaluated_directly_not_mirrored": True,
            "all_frequencies_finite": bool(np.isfinite(frequencies).all()),
            "all_d003_rises_positive": bool(
                all(
                    row["KG_d003_rise_cm-1"] > 0.0
                    and row["KM_d003_rise_cm-1"] > 0.0
                    for row in metrics
                )
            ),
            "finite_K_anchors_exact": bool(
                np.allclose(
                    finite_frequency[:, 0],
                    anchors["K_frequency_cm1"],
                    atol=1.0e-10,
                    rtol=0.0,
                )
            ),
        },
        "sources": {
            "zero_curve": str(args.zero_curve.resolve()),
            "finite_nk720_replay": str(args.finite_replay.resolve()),
            "Wannier_EPC": str(args.epmatwp.resolve()),
            "finite_short_range": "S0 L0 classical TDEP minus q6 EPC operator",
        },
        "limitations": [
            (
                "finite curves pair electronic smearings with 300/450/600 K "
                "lattice backgrounds; absolute vertical shifts are not a fixed-"
                "lattice-temperature smearing sweep"
            ),
            (
                "K-to-M finite curves are full-EPC model predictions without "
                "independent dense DFPT points on that direction"
            ),
            (
                "zero-smearing numerical regulator and convergence evidence "
                "remain those recorded in E14/E15"
            ),
        ],
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    summary_path = output / "zero_finite_full_kpath_Aprime_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
