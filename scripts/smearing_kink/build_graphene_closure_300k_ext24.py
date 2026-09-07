"""Build the frozen 300 K ext24 tagged-pair snapshots (12-seed -> 24-seed reference).

Motivation, frozen BEFORE any new DFT label is read (2026-09-08): the 12-seed
300 K reference failed the primary gate at 7.86 cm-1, diagnosed as
reference-statistics limited -- the amplitude-contiguous calib/heldout halves
disagreed by 15.77 cm-1 (the high-|A'| half is 1.81 eV/A^2 softer than the
low-|A'| half, per-seed std 1.57 eV/A^2 the largest of the three temperatures).
Weekly 2026-09-07 section 6 item 3 pre-priced this extension. Design:

  seeds  = 24 evenly spaced over the SAME amplitude-sorted 300-config
           formal_T300 ensemble (step = 300/24 = 12.5 instead of 25).  The
           frozen 12-seed picks are exactly every other pick (asserted), so
           all 24 existing labels are reused and only 12 NEW pairs are run.
  split  = amplitude-block interleaved halves: rank r = 0..23 in amplitude
           order; calibrate r % 4 in {0,1}, held out r % 4 in {2,3}.  The
           reused seeds sit on even ranks, so each half = 6 reused + 6 new
           seeds spanning the full amplitude range -- removing the amplitude
           imbalance that made the contiguous halves disagree by 15.77 cm-1.
  P      = unchanged: the deployed S0@300 short K-A' pattern, asserted
           bit-identical to the frozen 12-seed npz (same deployed inputs).

Pre-registered statistics (recorded before DFT): per-seed std ~1.57 eV/A^2 ->
12-seed half-mean SEM ~3.9 cm-1, interleaved split-half ~5.6 cm-1, and the
primary residual |K_ref(calib12) - K_ref(24)| = split-half/2 ~2.8 cm-1
expected scale (the contiguous 12-seed design realized 7.86).

Frozen gates for the ext24 readout:
  primary  |K_corrected - K_ref(24 seeds)| <= 5 cm-1
  heldout  |K_corrected - K_ref(held-out 12)| vs the interleaved split-half
           spread -- consistency readout, not pass/fail
  cusp     corrected cusp depths (d = 0.003/0.025) within 15% of the
           deployed S0@300 depths
  in-force primary AND cusp pass -> the ext24 dPi_calib goes IN FORCE at
           300 K; on failure the deployment stays un-recalibrated (measured
           bias carried) and the 12-seed measurement is kept as-is.
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
TEMPERATURE = 300
N_SEEDS_EXT = 24
MASS_C_AMU = 12.0107
DELTA_A = 0.04


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build() -> int:
    old_dir = BASE / f"R2B_{TEMPERATURE}k_tagged_pairs"
    outdir = BASE / f"R2B_{TEMPERATURE}k_tagged_pairs_ext24"
    outdir.mkdir(parents=True, exist_ok=True)
    old_npz = old_dir / "r2b_tagged_pairs_snapshots.npz"
    old_manifest = json.loads((old_dir / "r2b_pair_manifest.json").read_text())
    old_seeds = [int(s) for s in old_manifest["seeds"]]
    old_labels = sorted((old_dir / "labels").glob("snapshot_*_k8.npz"))
    if len(old_labels) != 2 * len(old_seeds):
        raise ValueError(f"expected {2 * len(old_seeds)} reused labels, found {len(old_labels)}")

    with np.load(fitter.DEPLOYED, allow_pickle=False) as data:
        qpoints = np.asarray(data["qpoints"], float)
        signed_q = np.asarray(data["signed_q_2pi_over_a"], float)
        response = np.asarray(data["full_EPC_response_cm2"], float)
        short_fc_dep = np.asarray(data["short_force_constants_eV_A2"], float)
        models = [str(m) for m in data["short_model"]]
        temps = [int(t) for t in data["lattice_temperature_K"]]
    mi, ti = models.index("S0"), temps.index(TEMPERATURE)
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

    # P must be bit-identical to the frozen 12-seed design (same deployed inputs)
    with np.load(old_npz, allow_pickle=False) as data:
        old_pattern = np.asarray(data["k_aprime_pattern"], float)
        old_pattern_norm = float(data["pattern_norm2"].reshape(()))
        old_scale = float(data["scale_cm2"].reshape(()))
        old_pi_k = float(data["pi_K_cm2"].reshape(()))
    if not (
        np.array_equal(pattern, old_pattern)
        and pattern_norm == old_pattern_norm
        and scale == old_scale
        and pi_k == old_pi_k
    ):
        raise ValueError("P/scale/pi_K drifted from the frozen 12-seed design")

    # --- seeds: 24 evenly spaced over the FULL formal_T300 ensemble ---
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
    xats = np.load(R2AP / f"formal_T{TEMPERATURE}/ensembles/xats_pop1.npy")[
        :, cc_to_phonopy, :
    ]
    displacements = displacement(xats)
    amplitudes = {
        int(i): float((pattern * displacements[i]).sum() / pattern_norm)
        for i in range(len(displacements))
    }
    order = sorted(range(len(displacements)), key=lambda i: abs(amplitudes[i]))
    step = len(order) / N_SEEDS_EXT
    seeds24 = []
    cursor = 0.0
    while len(seeds24) < N_SEEDS_EXT and int(cursor) < len(order):
        candidate = order[int(cursor)]
        if candidate not in seeds24:
            seeds24.append(candidate)
        cursor += step
    if len(seeds24) != N_SEEDS_EXT:
        raise ValueError("could not select 24 seeds")
    if not set(old_seeds) <= set(seeds24):
        raise ValueError("frozen 12-seed picks are not a subset of the 24 picks")
    reused_ranks = [r for r, s in enumerate(seeds24) if s in set(old_seeds)]
    if reused_ranks != list(range(0, 24, 2)):
        raise ValueError(f"reused seeds not on even ranks: {reused_ranks}")

    # amplitude-block interleaved halves: r%4 in {0,1} calibrate, {2,3} held out
    calib = [s for r, s in enumerate(seeds24) if r % 4 < 2]
    heldout = [s for r, s in enumerate(seeds24) if r % 4 >= 2]
    for half in (calib, heldout):
        n_reused = sum(1 for s in half if s in set(old_seeds))
        if len(half) != 12 or n_reused != 6:
            raise ValueError("interleaved halves must be 6 reused + 6 new each")

    # only the NEW seeds need DFT; ids are local to this extension tree
    new_seeds = [s for s in seeds24 if s not in set(old_seeds)]
    positions = []
    label_ids = []
    pair_signs = []
    for i, seed in enumerate(new_seeds):
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
        seed_sscha_indices=np.asarray(new_seeds, int),
        pair_sign=np.asarray(pair_signs, int),
        delta_A=np.full(len(positions), DELTA_A),
        k_aprime_pattern=pattern,
        pattern_norm2=np.array(pattern_norm),
        deployed_short_omega_K_cm1=np.array(omega_short_s0),
        scale_cm2=np.array(scale),
        pi_K_cm2=np.array(pi_k),
        lattice_temperature_K=np.array(float(TEMPERATURE)),
    )

    payload = {
        "status": "frozen_before_DFT",
        "purpose": (
            "enlarge the 300 K tagged-pair curvature reference from 12 to 24 seeds "
            "to close the primary gate (7.86 -> <=5); weekly 2026-09-07 section 6 "
            "item 3, pre-priced at one V100 overnight"
        ),
        "lattice_temperature_K": TEMPERATURE,
        "extends": {
            "frozen_12seed_manifest": str(old_dir / "r2b_pair_manifest.json"),
            "frozen_12seed_manifest_sha256": sha256(old_dir / "r2b_pair_manifest.json"),
            "frozen_12seed_snapshots_sha256": sha256(old_npz),
            "reused_seeds": old_seeds,
            "reused_labels_available": len(old_labels),
            "P_identity_asserted": (
                "k_aprime_pattern bit-identical to the frozen 12-seed npz "
                "(same deployed S0@300 short Hessian inputs)"
            ),
        },
        "seed_rule": (
            "sort the full 300-config formal_T300 ensemble by |A' projected amplitude| "
            "with the deployed S0@300 short K pattern; take evenly spaced 24 "
            "(step 12.5).  The frozen 12-seed picks (step 25) are exactly every "
            "other pick -- asserted, so all 24 existing labels are reused"
        ),
        "seeds_amplitude_rank_order": seeds24,
        "new_seeds_dft_required": new_seeds,
        "calib_split": {
            "rule": "amplitude rank r: calibrate r%4 in {0,1}, held out r%4 in {2,3}",
            "calib_12": calib,
            "heldout_12": heldout,
            "balance": "6 reused + 6 new seeds per half, both spanning the full amplitude range",
            "motivation": (
                "the 12-seed contiguous halves differed by 15.77 cm-1 (high-|A'| half "
                "1.81 eV/A^2 softer); interleaving removes the amplitude imbalance"
            ),
        },
        "seed_Aprime_amplitudes_A": {str(s): amplitudes[s] for s in seeds24},
        "delta_A": DELTA_A,
        "label_id_encoding": "1000 + 2*i + (0 for minus, 1 for plus), i over NEW seeds only",
        "pattern": {
            "source": f"Gamma-eigenspace K cluster of the deployed S0@{TEMPERATURE} short Hessian",
            "eigenvalue_cluster_size": int(len(cluster)),
            "pattern_omega_cm1": pattern_omega,
            "b0_omega_cm1": omega_short_s0,
            "eigen_residual_RMS": eigen_residual,
        },
        "pre_registered_statistics": (
            "per-seed std ~1.57 eV/A^2 -> 12-seed half-mean SEM ~3.9 cm-1, interleaved "
            "split-half ~5.6 cm-1, primary = |K_ref(calib12) - K_ref(24)| = split-half/2 "
            "~2.8 cm-1 expected scale (contiguous 12-seed design realized 7.86)"
        ),
        "frozen_gates": {
            "primary": "|K_corrected - K_ref(24 seeds)| <= 5 cm-1",
            "heldout": (
                "|K_corrected - K_ref(held-out 12)| vs the interleaved split-half "
                "spread (consistency readout, not pass/fail)"
            ),
            "cusp": "corrected cusp depths (d=0.003/0.025) within 15% of deployed S0@300",
            "in_force_rule": (
                "primary AND cusp pass -> ext24 dPi_calib IN FORCE at 300 K; on failure "
                "the deployment stays un-recalibrated and the 12-seed measurement stands"
            ),
        },
        "inputs": {
            "deployed_matrices": str(fitter.DEPLOYED),
            "deployed_matrices_sha256": sha256(fitter.DEPLOYED),
            "operator": str(fitter.OPERATOR),
            "operator_sha256": sha256(fitter.OPERATOR),
            "ensemble": str(R2AP / f"formal_T{TEMPERATURE}/ensembles/xats_pop1.npy"),
            "atom_order_interface": "R2AT_s0_shared_initial_sensitivity/formal_T450_Tel300 acceptance (fixed cell property)",
        },
        "outputs": {"r2b_tagged_pairs_snapshots_sha256": sha256(snapshot)},
    }
    (outdir / "r2b_pair_ext24_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"T={TEMPERATURE}K ext24: pattern omega {pattern_omega:.4f} cm-1, cluster {len(cluster)}, residual {eigen_residual:.2e}")
    print(f"seeds24 (amplitude rank order) {seeds24}")
    print(f"new seeds (DFT required, {len(new_seeds)}) {new_seeds}")
    print(f"calib_12 {calib}")
    print(f"heldout_12 {heldout}")
    ampls = [amplitudes[s] for s in seeds24]
    print(f"|A'| range {min(abs(a) for a in ampls):.4f}..{max(abs(a) for a in ampls):.4f} A")
    print(f"wrote {snapshot} ({len(positions)} configs)")
    print(f"wrote {outdir / 'r2b_pair_ext24_manifest.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    return build()


if __name__ == "__main__":
    raise SystemExit(main())
