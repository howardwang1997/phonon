#!/usr/bin/env python3
"""Run the P0 graphene MLIP+long-range validation without target leakage.

Two independently reported datasets are produced:

* ``8x8`` is the main finite-displacement result.  DFT smearings 0.005, 0.04
  and 0.08 Ry determine the long-range envelope; 0.01 and 0.02 Ry are retained
  as development evidence, while 0.015, 0.03 and 0.06 Ry are the later
  prospective holdouts that were generated only after the law was frozen.
* ``15pt`` is the denser 6x6 diagnostic scan.  Only 0.01, 0.02, 0.04 and
  0.08 Ry determine the envelope; the other eleven points diagnose interpolation
  and extrapolation behavior but are not relabelled as prospective evidence.

Every displayed MLIP result is obtained by a finite-displacement calculation
through the real v11 MACE calculator plus the analytic long-range correction.
No DFT target is copied into the prediction arrays.  The script caches every
force-constant calculation, so it is safe to restart after interruption.

Examples (on the 2060 host)::

    conda run --no-capture-output -n phonon python \
      scripts/smearing_kink/graphene_p0_deploy.py --dataset 8x8 --device cuda
    conda run --no-capture-output -n phonon python \
      scripts/smearing_kink/graphene_p0_deploy.py --dataset 15pt --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import friedel_module as fm  # noqa: E402
from friedel_calc import (  # noqa: E402
    FriedelCorrection,
    FriedelMACECalculator,
    fc2_from_calc,
)
from phonon_accel.phonons import phonopy_to_ase  # noqa: E402

CM = 33.35641
K_PER_RY = 157887.0
Q_GAMMA = np.array([0.0, 0.0, 0.0])
Q_K = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])


@dataclass(frozen=True)
class Dataset:
    name: str
    data_dir: Path
    pattern: str
    reference_dg: float
    background_dg: float
    anchor_dgs: tuple[float, ...]
    prospective_dgs: tuple[float, ...]
    rmax: float


DATASETS = {
    "8x8": Dataset(
        name="8x8",
        data_dir=ROOT / "results" / "sc_conv",
        pattern="graphene_sc8_dg*_phonopy.yaml",
        reference_dg=0.005,
        background_dg=0.08,
        anchor_dgs=(0.005, 0.04, 0.08),
        prospective_dgs=(0.015, 0.03, 0.06),
        rmax=9.8,
    ),
    "15pt": Dataset(
        name="15pt",
        data_dir=ROOT / "results" / "graphene_kohn_fd",
        pattern="graphene_sc6_dg*_phonopy.yaml",
        reference_dg=0.01,
        background_dg=0.08,
        anchor_dgs=(0.01, 0.02, 0.04, 0.08),
        prospective_dgs=(),
        rmax=12.0,
    ),
}


def same_dg(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=0.0, atol=5e-7))


def dg_slug(dg: float) -> str:
    return f"{dg:.6f}".rstrip("0").rstrip(".")


def discover_yamls(cfg: Dataset) -> dict[float, Path]:
    found: dict[float, Path] = {}
    rx = re.compile(r"_dg([0-9.]+)_phonopy\.yaml$")
    for path in sorted(cfg.data_dir.glob(cfg.pattern)):
        match = rx.search(path.name)
        if match:
            found[float(match.group(1))] = path
    return dict(sorted(found.items()))


def require_point(mapping: dict[float, Path], target: float) -> float:
    for value in mapping:
        if same_dg(value, target):
            return value
    raise FileNotFoundError(f"missing DFT point dg={target:g}")


def log_interp_law(temperatures: np.ndarray, values: np.ndarray):
    order = np.argsort(temperatures)
    xp = np.log(np.asarray(temperatures, float)[order])
    fp = np.asarray(values, float)[order]

    def law(temperature: float) -> float:
        return float(np.interp(np.log(float(temperature)), xp, fp))

    return law


def unified_anchor_laws(anchor_keys, fitted):
    """Pre-existing cross-material healing shape, scaled by only three anchors.

    The exponents p=4.44, q=3.00 come from ``unified_kink_law`` and predate the
    prospective 8x8 targets.  Tstar is fixed by the interior anchor's fitted B;
    kappa uses the same fixed p to rise from zero to the interior anchor and
    fall back to zero at the background.  No non-anchor target enters either law.
    """
    if len(anchor_keys) != 3:
        raise ValueError("unified_anchor law requires exactly three anchors")
    ref_dg, mid_dg, bg_dg = sorted(anchor_keys)
    t_ref, t_mid, t_bg = np.asarray([ref_dg, mid_dg, bg_dg]) * K_PER_RY
    b_mid, k_mid, _ = fitted[mid_dg]
    p, q = 4.44, 3.00

    def normalized_shape(temperature, tstar):
        numerator = max(0.0, 1.0 - (float(temperature) / tstar) ** p)
        denominator = max(1e-15, 1.0 - (t_ref / tstar) ** p)
        return (numerator / denominator) ** q

    lo, hi = t_mid * (1.0 + 1e-10), 1.0e6
    for _ in range(100):
        trial = 0.5 * (lo + hi)
        if normalized_shape(t_mid, trial) < b_mid:
            lo = trial
        else:
            hi = trial
    tstar = 0.5 * (lo + hi)

    def b_law(temperature):
        temperature = float(temperature)
        if temperature <= t_ref:
            return 1.0
        if temperature >= t_bg:
            return 0.0
        return normalized_shape(temperature, tstar)

    def k_law(temperature):
        temperature = float(temperature)
        if temperature <= t_ref or temperature >= t_bg:
            return 0.0
        if temperature <= t_mid:
            reduced = (temperature - t_ref) / (t_mid - t_ref)
        else:
            reduced = (t_bg - temperature) / (t_bg - t_mid)
        return float(k_mid) * max(0.0, reduced) ** p

    metadata = {"p": p, "q": q, "Tstar_K": float(tstar),
                "B_mid": float(b_mid), "kappa_mid_invA": float(k_mid)}
    return b_law, k_law, metadata


def band_and_metrics(ph, fc: np.ndarray) -> dict[str, object]:
    """Return bands plus explicitly unit-labelled anomaly observables."""
    dist, freq_thz, label_pos, labels = fm.band_from_phonopy(
        ph, fc, path="MGKM", npoints=201
    )
    kink_k, _, kink_g = fm.kink_of(ph, fc, path="MGKM")
    ph.force_constants = fc
    ph.run_qpoints([Q_GAMMA, Q_K], with_dynamical_matrices=False)
    qfreq_cm = np.sort(
        np.asarray(ph.get_qpoints_dict()["frequencies"], float), axis=1
    ) * CM
    return {
        "dist": np.asarray(dist, float),
        "bands_cm": np.asarray(freq_thz, float) * CM,
        "label_pos": np.asarray(label_pos, float),
        "labels": np.asarray(labels),
        # high_sym_kinks defines these as THz per reciprocal-path distance.
        "kink_k_raw": float(kink_k),
        "kink_g_raw": float(kink_g),
        "gamma_e2g_cm": float(qfreq_cm[0, -1]),
        "k_ito_cm": float(qfreq_cm[1, -1]),
        "k_branches_cm": qfreq_cm[1],
    }


def save_fc(path: Path, fc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, fc2=np.asarray(fc))
    os.replace(tmp, path)


def load_fc(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            return np.asarray(data["fc2"])
    except Exception as exc:
        print(f"# ignoring unreadable cache {path}: {exc}", flush=True)
        return None


CSV_FIELDS = [
    "dataset",
    "dg_Ry",
    "T_el_K",
    "split",
    "B",
    "kappa_invA",
    "anchor_fit_rel_resid",
    "dft_kink_K_THz_per_q",
    "method_kink_K_THz_per_q",
    "abs_kink_K_error_THz_per_q",
    "dft_kink_G_THz_per_q",
    "method_kink_G_THz_per_q",
    "dft_K_iTO_cm-1",
    "method_K_iTO_cm-1",
    "dft_G_E2g_cm-1",
    "method_G_E2g_cm-1",
    "full_band_MAE_cm-1",
]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def is_anchor(cfg: Dataset, dg: float) -> bool:
    return any(same_dg(dg, anchor) for anchor in cfg.anchor_dgs)


def split_label(cfg: Dataset, dg: float) -> str:
    if is_anchor(cfg, dg):
        return "anchor"
    if any(same_dg(dg, value) for value in cfg.prospective_dgs):
        return "prospective_holdout"
    if cfg.name == "8x8":
        # These points were strict holdouts in the first P0 pass, but became
        # development evidence once their errors were inspected and used to
        # justify the prospective campaign.  Never relabel them as pristine.
        return "development_holdout"
    return "heldout"


def run_dataset(cfg: Dataset, base, args) -> None:
    paths = discover_yamls(cfg)
    if cfg.name == "15pt" and len(paths) != 15:
        raise RuntimeError(
            f"{cfg.name}: expected 15 DFT YAMLs, found {len(paths)} in "
            f"{cfg.data_dir}"
        )
    if cfg.name == "8x8":
        # The original five points plus any later prospective points are valid;
        # only anchor_dgs enter the law, so adding targets cannot change it.
        for required in (0.005, 0.01, 0.02, 0.04, 0.08):
            require_point(paths, required)
    ref_dg = require_point(paths, cfg.reference_dg)
    bg_dg = require_point(paths, cfg.background_dg)
    anchor_keys = [require_point(paths, value) for value in cfg.anchor_dgs]

    print(
        f"# {cfg.name}: {len(paths)} points; anchors="
        f"{[dg_slug(x) for x in anchor_keys]}; "
        f"prospective={[dg_slug(x) for x in paths if split_label(cfg, x) == 'prospective_holdout']}; "
        f"other_holdouts={[dg_slug(x) for x in paths if split_label(cfg, x) not in ('anchor', 'prospective_holdout')]}",
        flush=True,
    )
    ph_by_dg = {dg: fm.load_ph(path) for dg, path in paths.items()}
    fc_dft = {dg: ph.force_constants.copy() for dg, ph in ph_by_dg.items()}
    ph = ph_by_dg[bg_dg]
    tabs, _ = fm.pair_table(ph)
    d0_rows = fm.template_delta(fc_dft[ref_dg], fc_dft[bg_dg], tabs)

    # Only anchor points enter B(T_el), kappa(T_el).  Endpoints define the
    # sharp-template and melted-background limits; interior anchors are fitted.
    kappas = np.linspace(0.0, 2.0, 321)
    fitted: dict[float, tuple[float, float, float]] = {}
    for dg in anchor_keys:
        if same_dg(dg, ref_dg):
            fitted[dg] = (1.0, 0.0, 0.0)
        elif same_dg(dg, bg_dg):
            fitted[dg] = (0.0, 0.0, 0.0)
        else:
            fitted[dg] = fm.fit_template_env(
                fc_dft[dg],
                fc_dft[bg_dg],
                tabs,
                d0_rows,
                1.0,
                cfg.rmax,
                kappas,
            )
        b, kappa, resid = fitted[dg]
        print(
            f"# {cfg.name} anchor dg={dg_slug(dg)}: "
            f"B={b:.6f} kappa={kappa:.6f} rel_resid={resid:.6f}",
            flush=True,
        )

    anchor_t = np.asarray(anchor_keys) * K_PER_RY
    if args.law == "log_interp":
        b_law = log_interp_law(anchor_t, np.asarray([fitted[x][0] for x in anchor_keys]))
        k_law = log_interp_law(anchor_t, np.asarray([fitted[x][1] for x in anchor_keys]))
        law_metadata = {"interpolation_axis": "log_T"}
    elif args.law == "unified_anchor":
        if cfg.name != "8x8":
            raise ValueError("unified_anchor is currently defined only for the 8x8 audit")
        b_law, k_law, law_metadata = unified_anchor_laws(anchor_keys, fitted)
    else:
        raise ValueError(f"unknown law {args.law}")
    print(f"# {cfg.name}: frozen law={args.law} metadata={law_metadata}", flush=True)
    corr = FriedelCorrection(
        ph,
        fc_dft[bg_dg],
        fc_dft[ref_dg],
        rmin=1.0,
        rmax=cfg.rmax,
        B_law=b_law,
        kappa_law=k_law,
    )
    ref_atoms = phonopy_to_ase(ph.supercell)

    outdir = args.outdir
    cache = outdir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"graphene_{cfg.name}_holdout.csv"

    backbone_path = cache / f"graphene_{cfg.name}_v11_backbone_fc2.npz"
    fc_backbone = None if args.force else load_fc(backbone_path)
    if fc_backbone is None:
        print(f"# {cfg.name}: computing real v11 MACE backbone fc2", flush=True)
        _, fc_backbone = fc2_from_calc(
            ph, base, distance=args.distance, subtract_ref=True
        )
        save_fc(backbone_path, fc_backbone)
    else:
        print(f"# {cfg.name}: reuse {backbone_path}", flush=True)
    backbone_metrics = band_and_metrics(ph, fc_backbone)
    print(
        f"# {cfg.name}: bare MACE kink_K="
        f"{backbone_metrics['kink_k_raw']:.6f} THz/q-distance",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    dft_bands, method_bands = [], []
    dft_k_branches, method_k_branches = [], []
    started = time.monotonic()
    shared_axis = None

    for index, (dg, _) in enumerate(paths.items(), 1):
        temperature = dg * K_PER_RY
        dft = band_and_metrics(ph_by_dg[dg], fc_dft[dg])
        pred_path = cache / f"graphene_{cfg.name}_dg{dg_slug(dg)}_p0_fc2.npz"
        fc_pred = None if args.force else load_fc(pred_path)
        if fc_pred is None:
            print(
                f"# {cfg.name} [{index}/{len(paths)}] deploy dg={dg_slug(dg)} "
                f"T_el={temperature:.0f} K",
                flush=True,
            )
            calc = FriedelMACECalculator(base, ref_atoms, corr, temperature)
            _, fc_pred = fc2_from_calc(
                ph, calc, distance=args.distance, subtract_ref=True
            )
            save_fc(pred_path, fc_pred)
        else:
            print(
                f"# {cfg.name} [{index}/{len(paths)}] reuse dg={dg_slug(dg)} cache",
                flush=True,
            )
        pred = band_and_metrics(ph, fc_pred)
        band_mae = float(
            np.mean(np.abs(np.asarray(pred["bands_cm"]) - np.asarray(dft["bands_cm"])))
        )
        b_value, k_value = b_law(temperature), k_law(temperature)
        anchor_match = next((x for x in anchor_keys if same_dg(x, dg)), None)
        fit_resid = fitted[anchor_match][2] if anchor_match is not None else float("nan")
        row = {
            "dataset": cfg.name,
            "dg_Ry": f"{dg:.6g}",
            "T_el_K": f"{temperature:.3f}",
            "split": split_label(cfg, dg),
            "B": f"{b_value:.9g}",
            "kappa_invA": f"{k_value:.9g}",
            "anchor_fit_rel_resid": f"{fit_resid:.9g}",
            "dft_kink_K_THz_per_q": f"{dft['kink_k_raw']:.9g}",
            "method_kink_K_THz_per_q": f"{pred['kink_k_raw']:.9g}",
            "abs_kink_K_error_THz_per_q": f"{abs(pred['kink_k_raw'] - dft['kink_k_raw']):.9g}",
            "dft_kink_G_THz_per_q": f"{dft['kink_g_raw']:.9g}",
            "method_kink_G_THz_per_q": f"{pred['kink_g_raw']:.9g}",
            "dft_K_iTO_cm-1": f"{dft['k_ito_cm']:.9g}",
            "method_K_iTO_cm-1": f"{pred['k_ito_cm']:.9g}",
            "dft_G_E2g_cm-1": f"{dft['gamma_e2g_cm']:.9g}",
            "method_G_E2g_cm-1": f"{pred['gamma_e2g_cm']:.9g}",
            "full_band_MAE_cm-1": f"{band_mae:.9g}",
        }
        rows.append(row)
        write_rows(csv_path, rows)
        dft_bands.append(dft["bands_cm"])
        method_bands.append(pred["bands_cm"])
        dft_k_branches.append(dft["k_branches_cm"])
        method_k_branches.append(pred["k_branches_cm"])
        if shared_axis is None:
            shared_axis = dft
        print(
            f"# {cfg.name} dg={dg_slug(dg)} {row['split']}: "
            f"kink_K DFT={dft['kink_k_raw']:.4f}, method={pred['kink_k_raw']:.4f} "
            f"THz/q; band_MAE={band_mae:.2f} cm^-1; "
            f"elapsed={time.monotonic() - started:.0f}s",
            flush=True,
        )

    split = np.asarray([row["split"] for row in rows])
    heldout = split != "anchor"
    prospective = split == "prospective_holdout"
    development = split == "development_holdout"
    float_col = lambda name: np.asarray([float(row[name]) for row in rows])
    def masked_mean(values, mask):
        values = np.asarray(values, float)
        return float(np.mean(values[mask])) if np.any(mask) else float("nan")

    summary = {
        "dataset": cfg.name,
        "law": args.law,
        "law_metadata": law_metadata,
        "anchors_dg_Ry": [float(x) for x in anchor_keys],
        "holdouts_dg_Ry": [float(dg) for dg in paths if not is_anchor(cfg, dg)],
        "prospective_holdouts_dg_Ry": [
            float(dg) for dg in paths if split_label(cfg, dg) == "prospective_holdout"
        ],
        "n_anchor": int(np.sum(split == "anchor")),
        "n_heldout": int(np.sum(heldout)),
        "n_prospective_holdout": int(np.sum(prospective)),
        "heldout_kink_K_MAE_THz_per_q": masked_mean(
            float_col("abs_kink_K_error_THz_per_q"), heldout
        ),
        "heldout_K_iTO_MAE_cm-1": masked_mean(
            np.abs(float_col("method_K_iTO_cm-1") - float_col("dft_K_iTO_cm-1")),
            heldout,
        ),
        "heldout_G_E2g_MAE_cm-1": masked_mean(
            np.abs(float_col("method_G_E2g_cm-1") - float_col("dft_G_E2g_cm-1")),
            heldout,
        ),
        "heldout_full_band_MAE_cm-1": masked_mean(
            float_col("full_band_MAE_cm-1"), heldout
        ),
        "prospective_kink_K_MAE_THz_per_q": masked_mean(
            float_col("abs_kink_K_error_THz_per_q"), prospective
        ),
        "prospective_K_iTO_MAE_cm-1": masked_mean(
            np.abs(float_col("method_K_iTO_cm-1") - float_col("dft_K_iTO_cm-1")),
            prospective,
        ),
        "prospective_G_E2g_MAE_cm-1": masked_mean(
            np.abs(float_col("method_G_E2g_cm-1") - float_col("dft_G_E2g_cm-1")),
            prospective,
        ),
        "prospective_full_band_MAE_cm-1": masked_mean(
            float_col("full_band_MAE_cm-1"), prospective
        ),
        "development_kink_K_MAE_THz_per_q": masked_mean(
            float_col("abs_kink_K_error_THz_per_q"), development
        ),
        "bare_MACE_kink_K_THz_per_q": float(backbone_metrics["kink_k_raw"]),
        "elapsed_seconds": float(time.monotonic() - started),
    }
    np.savez_compressed(
        outdir / f"graphene_{cfg.name}_holdout.npz",
        dgs_Ry=np.asarray(list(paths), float),
        T_el_K=np.asarray(list(paths), float) * K_PER_RY,
        split=split,
        law=np.asarray(args.law),
        B=float_col("B"),
        kappa_invA=float_col("kappa_invA"),
        distances=np.asarray(shared_axis["dist"]),
        label_positions=np.asarray(shared_axis["label_pos"]),
        labels=np.asarray(shared_axis["labels"]),
        dft_bands_cm=np.asarray(dft_bands),
        method_bands_cm=np.asarray(method_bands),
        dft_K_branches_cm=np.asarray(dft_k_branches),
        method_K_branches_cm=np.asarray(method_k_branches),
        dft_kink_K_THz_per_q=float_col("dft_kink_K_THz_per_q"),
        method_kink_K_THz_per_q=float_col("method_kink_K_THz_per_q"),
        dft_K_iTO_cm=float_col("dft_K_iTO_cm-1"),
        method_K_iTO_cm=float_col("method_K_iTO_cm-1"),
        dft_G_E2g_cm=float_col("dft_G_E2g_cm-1"),
        method_G_E2g_cm=float_col("method_G_E2g_cm-1"),
        full_band_MAE_cm=float_col("full_band_MAE_cm-1"),
        backbone_bands_cm=np.asarray(backbone_metrics["bands_cm"]),
    )
    summary_path = outdir / f"graphene_{cfg.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"# {cfg.name} COMPLETE: {json.dumps(summary, sort_keys=True)}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("8x8", "15pt", "both"), default="both")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "results" / "gr_backbone_v11" / "ft_graphene.model",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--law", choices=("log_interp", "unified_anchor"),
                        default="log_interp")
    parser.add_argument("--distance", type=float, default=0.03)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "p0_graphene")
    parser.add_argument("--force", action="store_true", help="ignore all cached MLIP fc2 files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model = args.model.resolve()
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.model.is_file():
        raise FileNotFoundError(f"missing MACE model: {args.model}")

    from mace.calculators import MACECalculator

    print(f"# loading MACE model {args.model} on {args.device}", flush=True)
    base = MACECalculator(
        model_paths=str(args.model), device=args.device, default_dtype="float32"
    )
    selected = tuple(DATASETS) if args.dataset == "both" else (args.dataset,)
    for name in selected:
        run_dataset(DATASETS[name], base, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
