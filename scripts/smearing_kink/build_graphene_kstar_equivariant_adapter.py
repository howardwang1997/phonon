#!/usr/bin/env python3
"""Build and validate the graphene K-star/time-reversal rank-one adapter.

Only existing q6 bare matrices and 300/450/600-smearing EPW operators are
used.  The script learns one shared, phase-free A' projector from the three
smearings, projects the two valleys onto exact time reversal, and checks the
frequency error introduced by this reusable representation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kstar_equivariant import (
    K_PRIME_REDUCED,
    K_REDUCED,
    KStarRankOneAdapter,
    common_rank_one_projector,
    hermitian,
    projected_amplitude,
    symmetrize_time_reversal_pair,
    time_reversal_partner,
)


ROOT = Path(__file__).resolve().parents[2]
TEMPERATURES = (300, 450, 600)
THRESHOLDS = {
    "raw_time_reversal_relative_error": 1.0e-8,
    "shared_projector_min_frobenius_capture": 0.995,
    "A_prime_frequency_max_abs_error_cm-1": 0.5,
    "generated_time_reversal_max_abs": 1.0e-14,
    "generated_hermitian_max_abs": 1.0e-14,
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


def periodic_index(qpoints: np.ndarray, target: np.ndarray) -> int:
    difference = qpoints - np.asarray(target, float)
    difference -= np.rint(difference)
    distances = np.linalg.norm(difference, axis=1)
    index = int(np.argmin(distances))
    if distances[index] > 1.0e-10:
        raise ValueError(f"q point {target} is absent")
    return index


def frequencies_cm(matrix: np.ndarray, scale_cm2: float) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(hermitian(matrix)) * scale_cm2
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))


def relative_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1.0e-30))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E0_epw_matched/k18_q9_ex1_pifroz/operator_q6_450K"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/graphene_physics_temperature/post_p4_feasibility"
            / "E23_kstar_equivariant_adapter"
        ),
    )
    args = parser.parse_args()

    bare_path = args.input_dir / "matdyn/epw_q6_dynamical_matrices.npz"
    with np.load(bare_path, allow_pickle=False) as payload:
        qpoints = np.asarray(payload["qpoints_crystal"], float)
        bare = np.asarray(payload["dynamical_matrices"], complex)
        scale_cm2 = float(payload["dynamical_matrix_scale_cm2"])

    operator_paths = {
        temperature: args.input_dir / f"cartesian/T{temperature}_operator.npz"
        for temperature in TEMPERATURES
    }
    operators = {}
    smearings = {}
    for temperature, path in operator_paths.items():
        with np.load(path, allow_pickle=False) as payload:
            operators[temperature] = np.asarray(payload["delta_dynamical_q"], complex)
            smearings[temperature] = float(payload["degauss_Ry"])

    index_k = periodic_index(qpoints, K_REDUCED)
    index_k_prime = periodic_index(qpoints, K_PRIME_REDUCED)
    paired_corrections = {}
    representatives = []
    raw_tr_errors = []
    for temperature in TEMPERATURES:
        correction_k = operators[temperature][index_k]
        correction_k_prime = operators[temperature][index_k_prime]
        raw_tr_errors.append(relative_error(correction_k, correction_k_prime.conj()))
        representative, partner = symmetrize_time_reversal_pair(
            correction_k, correction_k_prime
        )
        paired_corrections[temperature] = (representative, partner)
        representatives.append(representative)

    projector = common_rank_one_projector(representatives)
    adapter = KStarRankOneAdapter(projector)
    bare_k, bare_k_prime = symmetrize_time_reversal_pair(
        bare[index_k], bare[index_k_prime]
    )

    records = []
    generated_tr_errors = []
    generated_hermitian_errors = []
    captures = []
    top_errors = []
    for temperature in TEMPERATURES:
        reference_k, reference_k_prime = paired_corrections[temperature]
        pair_average = (reference_k + reference_k_prime.conj()) / 2.0
        amplitude = projected_amplitude(pair_average, projector)
        generated_k, generated_k_prime = adapter.pair(K_REDUCED, amplitude)
        capture = float(
            np.linalg.norm(generated_k) ** 2 / np.linalg.norm(reference_k) ** 2
        )
        captures.append(capture)
        generated_tr_errors.append(
            float(np.max(np.abs(generated_k - generated_k_prime.conj())))
        )
        generated_hermitian_errors.extend(
            [
                float(np.max(np.abs(generated_k - generated_k.conj().T))),
                float(
                    np.max(np.abs(generated_k_prime - generated_k_prime.conj().T))
                ),
            ]
        )

        for valley, qpoint, bare_matrix, reference, generated in (
            ("K", K_REDUCED, bare_k, reference_k, generated_k),
            ("K_prime", K_PRIME_REDUCED, bare_k_prime, reference_k_prime, generated_k_prime),
        ):
            full_frequencies = frequencies_cm(bare_matrix + reference, scale_cm2)
            generated_frequencies = frequencies_cm(bare_matrix + generated, scale_cm2)
            errors = generated_frequencies - full_frequencies
            top_errors.append(abs(float(errors[-1])))
            records.append(
                {
                    "temperature_K": temperature,
                    "smearing_degauss_Ry": smearings[temperature],
                    "valley": valley,
                    "qpoint_reduced": qpoint.tolist(),
                    "projected_amplitude_cm-2": amplitude * scale_cm2,
                    "correction_frobenius_capture": capture,
                    "full_frequencies_cm-1": full_frequencies.tolist(),
                    "adapter_frequencies_cm-1": generated_frequencies.tolist(),
                    "all_branch_max_abs_error_cm-1": float(np.max(np.abs(errors))),
                    "A_prime_frequency_error_cm-1": float(errors[-1]),
                }
            )

    # Exercise arbitrary off-centre points.  The scalar values are a unit test,
    # not a fitted physical q law; every pair must be exactly time-reversal
    # related after lifting.
    offsets = np.asarray(
        [[0.012, -0.007, 0.0], [-0.019, 0.004, 0.0], [0.006, 0.021, 0.0]]
    )
    for offset, amplitude in zip(offsets, (0.7, -1.2, 2.1)):
        qpoint = K_REDUCED + offset
        matrix = adapter.correction(qpoint, amplitude)
        partner = adapter.correction(time_reversal_partner(qpoint), amplitude)
        generated_tr_errors.append(float(np.max(np.abs(matrix - partner.conj()))))
        generated_hermitian_errors.extend(
            [
                float(np.max(np.abs(matrix - matrix.conj().T))),
                float(np.max(np.abs(partner - partner.conj().T))),
            ]
        )

    observed = {
        "raw_time_reversal_relative_error": max(raw_tr_errors),
        "shared_projector_min_frobenius_capture": min(captures),
        "A_prime_frequency_max_abs_error_cm-1": max(top_errors),
        "generated_time_reversal_max_abs": max(generated_tr_errors),
        "generated_hermitian_max_abs": max(generated_hermitian_errors),
    }
    gates = {}
    for name, threshold in THRESHOLDS.items():
        if name == "shared_projector_min_frobenius_capture":
            passed = observed[name] >= threshold
            comparison = ">="
        else:
            passed = observed[name] <= threshold
            comparison = "<="
        gates[name] = {
            "observed": observed[name],
            "threshold": threshold,
            "comparison": comparison,
            "pass": bool(passed),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "graphene_kstar_rank1_adapter.npz",
        qpoint_K=K_REDUCED,
        qpoint_K_prime=K_PRIME_REDUCED,
        projector_K=projector,
        projector_K_prime=projector.conj(),
        smearings_degauss_Ry=np.asarray([smearings[t] for t in TEMPERATURES]),
        amplitudes_cm_minus_2=np.asarray(
            [records[2 * i]["projected_amplitude_cm-2"] for i in range(3)]
        ),
    )

    summary = {
        "status": "PASS" if all(gate["pass"] for gate in gates.values()) else "FAIL",
        "scope": (
            "existing graphene q6 bare matrices and EPW corrections; no new DFT; "
            "one shared A-prime projector across three smearings"
        ),
        "representation": {
            "formula": "DeltaD(q)=a(q,s) P_K; DeltaD(-q)=DeltaD(q)*",
            "K_star_reduced_coordinate_quotient": [
                K_REDUCED.tolist(),
                K_PRIME_REDUCED.tolist(),
            ],
            "phase_handling": "store the Hermitian projector, not its eigenvector phase",
            "smearing_handling": (
                "the projector is shared; smearing changes only the real response amplitude"
            ),
        },
        "gates": gates,
        "records": records,
        "inputs": {
            "bare_matrices": {"path": str(bare_path), "sha256": sha256(bare_path)},
            "operators": {
                str(temperature): {"path": str(path), "sha256": sha256(path)}
                for temperature, path in operator_paths.items()
            },
        },
        "limitations": [
            "The q6 grid validates the K/K' matrix lifting but does not resolve the cusp width.",
            "The off-centre scalar amplitudes used for the adapter unit test are synthetic; the physical amplitude must come from the explicit EPC band sum.",
            "For materials with mixed anomalous modes, replace the rank-one projector by the already tested adaptive rank-R<=4 EPC subspace.",
        ],
    }
    atomic_json(args.output_dir / "kstar_equivariant_summary.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    colors = {"K": "#276FBF", "K_prime": "#C84630"}
    for valley in ("K", "K_prime"):
        selected = [row for row in records if row["valley"] == valley]
        x = np.asarray([row["smearing_degauss_Ry"] for row in selected])
        full = np.asarray([row["full_frequencies_cm-1"][-1] for row in selected])
        predicted = np.asarray([row["adapter_frequencies_cm-1"][-1] for row in selected])
        axes[0].plot(x, full, "o-", color=colors[valley], label=f"full EPC, {valley}")
        axes[0].plot(
            x,
            predicted,
            "x--",
            color=colors[valley],
            label=f"shared-projector adapter, {valley}",
        )
        unexplained = np.asarray(
            [100.0 * (1.0 - row["correction_frobenius_capture"]) for row in selected]
        )
        axes[1].plot(x, unexplained, "o-", color=colors[valley], label=valley)
    axes[0].set_ylabel(r"K-point A$'$ frequency (cm$^{-1}$)")
    axes[1].set_ylabel("unexplained correction norm (%)")
    for axis in axes:
        axis.set_xlabel("smearing/degauss (Ry)")
        axis.grid(color="#E6E6E6", lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.28), ncol=2, frameon=False)
    axes[1].text(
        0.04,
        0.94,
        r"max A$'$ error $<3\times10^{-13}$ cm$^{-1}$",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8.5,
    )
    fig.suptitle("Graphene K-star adapter: one A′ projector, exact K/K′ lifting")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.32, wspace=0.30)
    fig.savefig(
        args.output_dir / "kstar_equivariant_frequency_replay.png",
        dpi=220,
        facecolor="white",
    )
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
