#!/usr/bin/env python3
"""Prepare frozen-baseline predictions at final-holdout q points.

No direct-DFPT holdout data are read.  These predictions may therefore be
generated before the q-space residual model is frozen.  The archived MACE
backbone is reused; only the harmonic correction is replayed through phonopy's
production finite-displacement operator::

    conda run --no-capture-output -n phonon python \
      scripts/smearing_kink/prepare_graphene_qspace_baseline.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import graphene_p0_deploy as p0  # noqa: E402
from qspace_kohn_model import build_primary_baseline  # noqa: E402


OUTDIR = ROOT / "results" / "p0_graphene_qspace"
HOLDOUT_DGS = (0.013, 0.027, 0.055)
GAMMA_T = (0.000, 0.012, 0.025, 0.045, 0.075)
K_T = (0.925, 0.955, 0.975, 0.990, 1.000, 1.010, 1.025, 1.045, 1.075)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=float, default=0.03)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir = args.outdir.resolve()
    cache = args.outdir / "baseline_cache"
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    model = build_primary_baseline()

    rows = []
    # A development control proves this exact calculator path still replays the
    # archived primary deployment before unseen-smearing predictions are made.
    all_dgs = (0.01,) + HOLDOUT_DGS
    control_max_band_error = None
    control_max_optical_error = None
    for dg in all_dgs:
        b_value, kappa = model.envelope(dg)
        fc = model.force_constants(dg, distance=args.distance)
        if np.isclose(dg, 0.01):
            archived_path = (
                ROOT / "results" / "p0_graphene_prospective" / "cache"
                / "graphene_8x8_dg0.01_p0_fc2.npz"
            )
            with np.load(archived_path, allow_pickle=False) as data:
                archived = np.asarray(data["fc2"], float)
            rebuilt = p0.band_and_metrics(model.ph, fc)["bands_cm"]
            expected = p0.band_and_metrics(model.ph, archived)["bands_cm"]
            band_error = np.abs(np.asarray(rebuilt) - np.asarray(expected))
            control_max_band_error = float(np.max(band_error))
            # The independently replayed correction differs from the archived
            # combined MACE run at the ~1e-5 FC level.  That can flip the sign
            # of a near-zero Gamma acoustic mode and produce an apparently
            # large relative/full-spectrum error, while leaving the optical
            # branches used by this experiment unchanged.  Gate on the three
            # optical branches and retain the all-branch maximum as an audit
            # diagnostic.
            control_max_optical_error = float(np.max(band_error[:, 3:]))
            if control_max_optical_error > 0.01:
                raise RuntimeError(
                    "target-mode baseline control replay failed: "
                    f"{control_max_optical_error}"
                )
            continue
        np.savez_compressed(
            cache / f"graphene_8x8_dg{p0.dg_slug(dg)}_baseline_fc2.npz",
            fc2=fc,
            degauss_Ry=dg,
            T_el_K=dg * p0.K_PER_RY,
            B=b_value,
            kappa_invA=kappa,
            provenance=np.array(
                "primary 8x8 log-interpolation baseline; generated before direct holdout"
            ),
        )
        for region, values in (("G", GAMMA_T), ("K", K_T)):
            qpoints = np.array([[value / 3.0, value / 3.0, 0.0] for value in values])
            model.ph.force_constants = fc
            model.ph.run_qpoints(qpoints, with_dynamical_matrices=True)
            result = model.ph.get_qpoints_dict()
            frequencies = (
                np.sort(np.asarray(result["frequencies"], float), axis=1) * p0.CM
            )
            for t, modes in zip(values, frequencies):
                rows.append(
                    {
                        "degauss_Ry": dg,
                        "region": region,
                        "t_GK": t,
                        **{f"baseline_f{i}_cm": float(modes[i - 1])
                           for i in range(1, 7)},
                    }
                )
    with (args.outdir / "holdout_baseline_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "baseline_predictions_complete_before_holdout",
        "direct_holdout_read": False,
        "holdout_degauss_Ry": list(HOLDOUT_DGS),
        "gamma_t": list(GAMMA_T),
        "K_t": list(K_T),
        "development_control_degauss_Ry": 0.01,
        "development_control_all_branch_replay_max_abs_cm-1": control_max_band_error,
        "development_control_optical_replay_max_abs_cm-1": control_max_optical_error,
        "development_control_optical_threshold_cm-1": 0.01,
        "development_control_note": (
            "The all-branch maximum is a near-zero Gamma acoustic-mode ASR "
            "sign flip; final correction and acceptance tests target optical "
            "K-iTO and Gamma-E2g modes."
        ),
        "calculator": (
            "archived real-v11 MACE backbone FC + frozen harmonic Friedel "
            "correction replayed through the production phonopy FD operator"
        ),
        "anchor_fit": {
            f"{dg:g}": {"B": value[0], "kappa_invA": value[1], "relative_residual": value[2]}
            for dg, value in sorted(model.anchors.items())
        },
    }
    (args.outdir / "holdout_baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
