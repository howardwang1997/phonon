"""Build the R2B tagged-pair snapshots for the 450 K closure reference.

The plain TDEP force fit cannot deliver a K A' absolute reference at useful
precision: a synthetic study on the stored R2AO forces showed the fitted K
eigenvalue scatters by ~60 cm-1 across 12-config subsets (the shard-only fit
that happened to reproduce the deployed value was luck), and even the direct
A'-force regression sits ~+30 cm-1 above the SSCHA eigenvalue with +-30 cm-1
subset noise, because a force regression estimates <V''> plus asymmetric
anharmonic terms rather than the SSCHA stationarity object <d^2V/da^2>_T.

The deployed thermal Hessian is the SSCHA free-energy Hessian, whose
stationary condition is exactly the thermal average of the local curvature.
R2B therefore measures that object directly: for each frozen-reserve seed
configuration u_i, two DFT labels are computed at u_i +- delta * P, where P
is the (verified) K A' eigendisplacement of the deployed S0@450 short
Hessian (Gamma-eigenspace cluster of the 6x6 supercell).  The projected
restoring curvature [f(+) - f(-)] / (2 delta), averaged over seeds after
subtracting the q6 operator force, is the independent thermal K A'
Hessian; adding the shared full-EPC Pi(K) gives the absolute K reference.

Seeds: 12 of the 36 frozen reserve indices, deterministic stratified rule
(sort by |A' projected amplitude|, take every third), mirror pairs excluded.
Label ids 1000 + 2*i + sign keep every tagged config uniquely addressable.
No DFT label is read by this script; the design is frozen before R1 returns.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import fit_graphene_450k_closure_tdep as fitter  # noqa: E402
import friedel_module as fm  # noqa: E402

BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
R2R = BASE / "R2R_multipolar_background"
OUT = BASE / "R1_450k_matched_overlap_screen"
MASS_C_AMU = 12.0107
DELTA_A = 0.04
N_SEEDS = 12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads((OUT / "freeze_manifest.json").read_text())

    with np.load(fitter.OPERATOR, allow_pickle=False) as data:
        operator = {key: np.asarray(data[key]) for key in data.files}
    reference = np.asarray(operator["reference_positions"], float)
    cell = np.asarray(operator["cell"], float)
    inv_cell = np.linalg.inv(cell)

    with np.load(fitter.DEPLOYED, allow_pickle=False) as data:
        qpoints = np.asarray(data["qpoints"], float)
        response = np.asarray(data["full_EPC_response_cm2"], float)
        short_fc_dep = np.asarray(data["short_force_constants_eV_A2"], float)
        full = np.asarray(data["full_EPC_frequency_cm1"], float)
        signed_q = np.asarray(data["signed_q_2pi_over_a"], float)
    k_index = int(np.argmin(np.abs(signed_q)))
    pi_k = float(response[k_index])

    # P: verified K A' eigendisplacement of the deployed S0@450 short Hessian.
    phonon = fm.load_ph(fitter.BACKGROUND)
    sequence = fitter.b0.dynamical_sequence(
        phonon, short_fc_dep[1, 1], qpoints[k_index : k_index + 1]
    )
    scale = float(sequence["scale_cm2"][0])
    omega_short_s0 = float(sequence["frequencies_cm1"][0, -1])
    lam_k = omega_short_s0**2 / scale
    matrix = short_fc_dep[1, 1].transpose(0, 2, 1, 3).reshape(216, 216) / MASS_C_AMU
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    pick = int(np.argmin(np.abs(values - lam_k)))
    cluster = np.flatnonzero(np.abs(values - values[pick]) < 1.0e-6)
    if not (1 <= len(cluster) <= 4):
        raise ValueError(f"unexpected K cluster size {len(cluster)}")
    pattern = vectors[:, pick].reshape(72, 3).copy()
    pattern_norm = float((pattern * pattern).sum())
    eigen_residual = float(
        np.sqrt(
            (
                np.abs(
                    np.einsum("ijab,jb->ia", short_fc_dep[1, 1], pattern)
                    - MASS_C_AMU * values[pick] * pattern
                )
                ** 2
            ).mean()
        )
    )
    pattern_omega = float(np.sqrt(values[pick] * scale))
    if abs(pattern_omega - omega_short_s0) > 1.0e-4 or eigen_residual > 1.0e-6:
        raise ValueError("K pattern failed the eigendisplacement check")

    def displacement(positions: np.ndarray) -> np.ndarray:
        delta = positions - reference
        fractional = delta @ inv_cell
        fractional -= np.round(fractional)
        return fractional @ cell

    # Seeds: stratified over the frozen reserve by |A' projected amplitude|.
    reserve = sorted(
        int(index)
        for batch in manifest["selection_protocol"]["reserved_R2A_batches"].values()
        for index in batch
    )
    if len(reserve) != 36:
        raise ValueError(f"expected 36 reserve indices, found {len(reserve)}")
    amplitudes = {}
    seed_displacements = {}
    for stem in ("batch_2", "batch_3", "batch_4"):
        with np.load(OUT / f"{stem}_snapshots.npz", allow_pickle=False) as data:
            positions_all = np.asarray(data["positions"], float)
            sscha_indices = np.asarray(data["sscha_indices"], int)
        for positions, index in zip(positions_all, sscha_indices):
            displacement_i = displacement(positions)
            amplitudes[int(index)] = float(
                (pattern * displacement_i).sum() / pattern_norm
            )
            seed_displacements[int(index)] = displacement_i
    order = sorted(reserve, key=lambda index: abs(amplitudes[index]))
    step = len(order) / N_SEEDS
    seeds = []
    cursor = 0.0
    while len(seeds) < N_SEEDS and int(cursor) < len(order):
        candidate = order[int(cursor)]
        if candidate not in seeds and (candidate ^ 1) not in seeds:
            seeds.append(candidate)
        cursor += step
    if len(seeds) != N_SEEDS:
        raise ValueError("could not select 12 mirror-free seeds")
    if any(index not in seed_displacements for index in seeds):
        raise ValueError("seed displacement missing")

    positions = []
    label_ids = []
    pair_signs = []
    for i, seed in enumerate(seeds):
        for sign in (-1, +1):
            tagged = seed_displacements[seed] + sign * DELTA_A * pattern
            positions.append(reference + tagged)
            label_ids.append(1000 + 2 * i + (0 if sign < 0 else 1))
            pair_signs.append(sign)
    positions = np.asarray(positions, float)
    if not np.isfinite(positions).all():
        raise ValueError("non-finite tagged positions")

    np.savez(
        OUT / "r2b_tagged_pairs_snapshots.npz",
        positions=positions,
        cells=np.repeat(cell[None, :, :], len(positions), axis=0),
        numbers=np.full(72, 6, dtype=int),
        sscha_indices=np.asarray(label_ids, int),
        selection_groups=np.array(["r2b_tagged_pair"] * len(positions), dtype="<U16"),
        seed_sscha_indices=np.asarray(seeds, int),
        pair_sign=np.asarray(pair_signs, int),
        delta_A=np.full(len(positions), DELTA_A),
        k_aprime_pattern=pattern,
        pattern_norm2=np.array(pattern_norm),
        deployed_short_omega_K_cm1=np.array(omega_short_s0),
        scale_cm2=np.array(scale),
        pi_K_cm2=np.array(pi_k),
    )

    payload = {
        "status": "frozen_before_R2B_DFT",
        "purpose": (
            "absolute K A' reference via tagged-pair thermal curvature; the plain "
            "TDEP fit retains only the shape/cusp gates"
        ),
        "seed_rule": (
            "sort the 36 frozen reserve indices by |A' projected amplitude| with the "
            "deployed S0@450 short K pattern; take evenly spaced 12, mirror pairs excluded"
        ),
        "seeds": seeds,
        "seed_Aprime_amplitudes_A": {
            str(seed): amplitudes[seed] for seed in seeds
        },
        "delta_A": DELTA_A,
        "label_id_encoding": "1000 + 2*i + (0 for minus, 1 for plus)",
        "pattern": {
            "source": "Gamma-eigenspace K cluster of the deployed S0@450 short Hessian",
            "eigenvalue_cluster_size": int(len(cluster)),
            "pattern_omega_cm1": pattern_omega,
            "b0_omega_cm1": omega_short_s0,
            "eigen_residual_RMS": eigen_residual,
        },
        "inputs": {
            "deployed_matrices_sha256": sha256(fitter.DEPLOYED),
            "operator_sha256": sha256(fitter.OPERATOR),
            "freeze_manifest": str(OUT / "freeze_manifest.json"),
        },
        "outputs": {
            "r2b_tagged_pairs_snapshots_sha256": sha256(
                OUT / "r2b_tagged_pairs_snapshots.npz"
            ),
        },
    }
    (OUT / "r2b_pair_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"seeds {seeds}")
    print(f"wrote {OUT / 'r2b_tagged_pairs_snapshots.npz'} ({len(positions)} configs)")
    print(f"wrote {OUT / 'r2b_pair_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
