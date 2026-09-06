"""Build frozen R2B tagged-pair snapshots for the 300/600 K replication.

Temperature-parametric generalization of ``build_graphene_450k_closure_r2b_pairs.py``
(same estimator, same verification chain), for the weekly §6 item "one curvature
reference per temperature":

    P     = verified K A' eigendisplacement of the deployed S0@T short Hessian
    seeds = 12 deterministic stratified picks from the FULL formal_T{T}
            300-config SSCHA ensemble (sort by |A' projected amplitude|,
            evenly spaced)
    labels = u_i +- 0.04 A * P, ids 1000 + 2*i + (0 for minus, 1 for plus)

Deviations from the 450 K design, both documented before any DFT is read:
  * seed population = the full ensemble (the 450 K builder sorted only the
    36-index R2A reserve, an artifact of the R1' screen that does not exist
    at other temperatures);
  * no mirror-pair exclusion: a nearest-match probe (max-projection cost over
    all pairs, 2026-09-07) shows these ensembles contain NO exact +- mirror
    pairs (best-match cost ~6 A = random-pair level, zero pairs < 1e-6), so
    the 450 K ``i ^ 1`` exclusion rule was vacuous.  Dropping it removes a
    no-op constraint; the frozen 450 K seeds are unaffected by this finding
    (none of the 12 clash as mutual mirrors).

Frozen gates for each temperature (same statistical hierarchy as 450 K):
  primary  |K_corrected - K_ref(12 seeds)| <= 5 cm-1
  heldout  |K_corrected - K_ref(held-out 6)| vs that temperature's split-half
           spread -- consistency readout, not pass/fail (a 6-seed reference's
           own statistical spread exceeds 5 cm-1)
  cusp     corrected cusp depths (d = 0.003 / 0.025) within 15% of the
           deployed S0@T depths
  readout  dPi_calib(T) vs the 450 K constant (+70261.6 cm-2): the
           transferability question this replication exists to answer.
No shape-vs-DFT gate: no plain TDEP labels at these temperatures (scope
reduction vs the 450 K closure, which established morphology).
"""
from __future__ import annotations

import argparse
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
from audit_graphene_current_s0_fixed_smearing import load_operator  # noqa: E402

BASE = ROOT / "results/graphene_physics_temperature/post_p4_feasibility"
R2AP = BASE / "R2R_multipolar_background/R2AP_fixed_smearing_SSCHA"
SHARED = BASE / "R2R_multipolar_background/R2AT_s0_shared_initial_sensitivity/formal_T450_Tel300"
MASS_C_AMU = 12.0107
DELTA_A = 0.04
N_SEEDS = 12
DPI_450K_CM2 = 70261.60368031562  # measured 2026-09-07, commit f23d33a


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(temperature: int) -> int:
    outdir = BASE / f"R2B_{temperature}k_tagged_pairs"
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(fitter.DEPLOYED, allow_pickle=False) as data:
        qpoints = np.asarray(data["qpoints"], float)
        signed_q = np.asarray(data["signed_q_2pi_over_a"], float)
        response = np.asarray(data["full_EPC_response_cm2"], float)
        short_fc_dep = np.asarray(data["short_force_constants_eV_A2"], float)
        models = [str(m) for m in data["short_model"]]
        temps = [int(t) for t in data["lattice_temperature_K"]]
    mi, ti = models.index("S0"), temps.index(temperature)
    short_fc = short_fc_dep[mi, ti]
    k_index = int(np.argmin(np.abs(signed_q)))
    pi_k = float(response[k_index])

    # --- P: verified K A' eigendisplacement of the deployed S0@T short Hessian ---
    phonon = fm.load_ph(fitter.BACKGROUND)
    sequence = fitter.b0.dynamical_sequence(
        phonon, short_fc, qpoints[k_index : k_index + 1]
    )
    scale = float(sequence["scale_cm2"][0])
    omega_short_s0 = float(sequence["frequencies_cm1"][0, -1])
    lam_k = omega_short_s0**2 / scale
    matrix = short_fc.transpose(0, 2, 1, 3).reshape(216, 216) / MASS_C_AMU
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
                    np.einsum("ijab,jb->ia", short_fc, pattern)
                    - MASS_C_AMU * values[pick] * pattern
                )
                ** 2
            ).mean()
        )
    )
    pattern_omega = float(np.sqrt(values[pick] * scale))
    if abs(pattern_omega - omega_short_s0) > 1.0e-4 or eigen_residual > 1.0e-6:
        raise ValueError("K pattern failed the eigendisplacement check")

    # --- seeds: stratified over the FULL formal_T{T} ensemble ---
    _, reference, cell, _ = load_operator(fitter.OPERATOR)
    inv_cell = np.linalg.inv(cell)

    def displacement(positions: np.ndarray) -> np.ndarray:
        delta = positions - reference
        fractional = delta @ inv_cell
        fractional -= np.round(fractional)
        return fractional @ cell

    acceptance = json.loads((SHARED / "acceptance.json").read_text())
    cc_to_phonopy = np.asarray(
        acceptance["atom_order_interface"]["CellConstructor_for_phonopy"], int
    )
    xats = np.load(R2AP / f"formal_T{temperature}/ensembles/xats_pop1.npy")[
        :, cc_to_phonopy, :
    ]
    displacements = displacement(xats)
    amplitudes = {
        int(i): float((pattern * displacements[i]).sum() / pattern_norm)
        for i in range(len(displacements))
    }
    order = sorted(range(len(displacements)), key=lambda i: abs(amplitudes[i]))
    step = len(order) / N_SEEDS
    seeds = []
    cursor = 0.0
    while len(seeds) < N_SEEDS and int(cursor) < len(order):
        candidate = order[int(cursor)]
        if candidate not in seeds:
            seeds.append(candidate)
        cursor += step
    if len(seeds) != N_SEEDS:
        raise ValueError("could not select 12 seeds")

    positions = []
    label_ids = []
    pair_signs = []
    for i, seed in enumerate(seeds):
        for sign in (-1, +1):
            tagged = displacements[seed] + sign * DELTA_A * pattern
            positions.append(reference + tagged)
            label_ids.append(1000 + 2 * i + (0 if sign < 0 else 1))
            pair_signs.append(sign)
    positions = np.asarray(positions, float)
    if not np.isfinite(positions).all():
        raise ValueError("non-finite tagged positions")

    snapshot = outdir / "r2b_tagged_pairs_snapshots.npz"
    np.savez(
        snapshot,
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
        lattice_temperature_K=np.array(float(temperature)),
    )

    payload = {
        "status": "frozen_before_DFT",
        "purpose": (
            f"absolute K A' reference via tagged-pair thermal curvature at "
            f"T_lat={temperature} K; second/third temperature point of the "
            "dPi_calib(T_lat) replication (weekly 2026-09-07 section 6)"
        ),
        "lattice_temperature_K": temperature,
        "seed_rule": (
            "sort the full 300-config formal_T ensemble by |A' projected amplitude| "
            f"with the deployed S0@{temperature} short K pattern; take evenly spaced 12"
        ),
        "mirror_exclusion": (
            "none: nearest-match probe 2026-09-07 shows the ensembles contain no "
            "exact +- mirror pairs (best-match cost ~6 A, zero < 1e-6); the 450 K "
            "i^1 rule was vacuous and is dropped"
        ),
        "seeds": seeds,
        "calib_split": {"calib_6": seeds[:6], "heldout_6": seeds[6:]},
        "seed_Aprime_amplitudes_A": {str(s): amplitudes[s] for s in seeds},
        "delta_A": DELTA_A,
        "label_id_encoding": "1000 + 2*i + (0 for minus, 1 for plus)",
        "pattern": {
            "source": f"Gamma-eigenspace K cluster of the deployed S0@{temperature} short Hessian",
            "eigenvalue_cluster_size": int(len(cluster)),
            "pattern_omega_cm1": pattern_omega,
            "b0_omega_cm1": omega_short_s0,
            "eigen_residual_RMS": eigen_residual,
        },
        "frozen_gates": {
            "primary": "|K_corrected - K_ref(12 seeds)| <= 5 cm-1",
            "heldout": (
                "|K_corrected - K_ref(held-out 6)| vs that temperature's split-half "
                "spread (consistency readout, not pass/fail)"
            ),
            "cusp": "corrected cusp depths (d=0.003/0.025) within 15% of deployed S0@T",
            "shape_vs_dft": (
                "not applicable: no plain TDEP labels at this temperature "
                "(scope reduction vs the 450 K closure)"
            ),
        },
        "transferability_readout": {
            "dpi_calib_450K_cm2": DPI_450K_CM2,
            "question": "does the 450 K calibration constant transfer, or is dPi_calib(T) needed per temperature",
        },
        "inputs": {
            "deployed_matrices": str(fitter.DEPLOYED),
            "deployed_matrices_sha256": sha256(fitter.DEPLOYED),
            "operator": str(fitter.OPERATOR),
            "operator_sha256": sha256(fitter.OPERATOR),
            "ensemble": str(R2AP / f"formal_T{temperature}/ensembles/xats_pop1.npy"),
            "atom_order_interface": "R2AT_s0_shared_initial_sensitivity/formal_T450_Tel300 acceptance (fixed cell property)",
        },
        "outputs": {"r2b_tagged_pairs_snapshots_sha256": sha256(snapshot)},
    }
    (outdir / "r2b_pair_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"T={temperature}K: pattern omega {pattern_omega:.4f} cm-1, cluster {len(cluster)}, residual {eigen_residual:.2e}")
    print(f"seeds {seeds}")
    ampls = [amplitudes[s] for s in seeds]
    print(f"|A'| range {min(abs(a) for a in ampls):.4f}..{max(abs(a) for a in ampls):.4f} A")
    print(f"wrote {snapshot} ({len(positions)} configs)")
    print(f"wrote {outdir / 'r2b_pair_manifest.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=int, required=True, choices=(300, 600))
    args = parser.parse_args()
    return build(args.temperature)


if __name__ == "__main__":
    raise SystemExit(main())
