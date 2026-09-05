"""(L)-channel TDEP through FriedelMACE = backbone MACE + Friedel long-range @ T_el.

Why: td_phonon.py uses the BARE v11 backbone, which is smearing-blind by design
(kink_K~0). The Kohn kink lives in the Friedel long-range term. To measure
kink_K(T_lat) (does the kink move with lattice temperature?) AND to be comparable
to DFT-MD-TDEP@dg0.005 (which carries the kink), MD must run through the FULL model:
backbone MACE + FriedelCorrection(dg0.080 backbone, dg0.005 template) @ T_el.

Geometry + fc2 are ALL loaded from the dg0.080 yaml (ph0) so ref_pos / dfc / MD atoms
share one atom ordering (required by _harm_EF: force ~ pos - ref_pos). Mirrors
deploy_8x8 / friedel_calc.main (6x6) + reuses td_phonon.sample_md/effective_fc2/band.

Run on 2060:
  conda run -n phonon python scripts/smearing_kink/td_phonon_friedel.py \
    --model results/gr_backbone_v11/ft_graphene.model \
    --bg   results/vq_kink6/graphene_sc6_dg0.080_phonopy.yaml \
    --ref  results/vq_kink6/graphene_sc6_dg0.005_phonopy.yaml \
    --friedel-tel 789 --temperatures 100,300,600 --tag graphene_ft_friedel
"""
from __future__ import annotations
import sys, time, warnings, argparse
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts"))
import friedel_module as fm
from friedel_calc import (
    FixedHarmonicCorrection,
    FriedelCorrection,
    FriedelMACECalculator,
)
from conditioned_mace import conditioned_short_calculator
from phonon_accel.phonons import phonopy_to_ase
from ase.calculators.mixing import SumCalculator
import td_common as tdc
import anomaly_locate as al
import td_phonon as tdp  # reuse sample_md / effective_fc2 / band_from_phonopy
CM = 33.35641

# graphene healing law (from deploy_8x8 / fit_friedel). At T_el=789: B=1.0, kappa=0
# => reconstructs the dg0.005 fc2 (kink present) on top of the dg0.080 backbone.
Tp = [789, 1579, 3158, 6316, 12631]; Bp = [1.0, 1.0, 0.982, 0.874, 0.0]; Kp = [0, 0, 0, 0.05, 0]
Blaw = lambda T: float(np.interp(T, Tp, Bp)); Klaw = lambda T: float(np.interp(T, Tp, Kp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--delta-model", default="")
    ap.add_argument("--delta-model-300", default="")
    ap.add_argument("--delta-model-600", default="")
    ap.add_argument("--bg", required=True, help="6x6 backbone fc2 yaml (dg0.080, kink-free)")
    ap.add_argument("--ref", default="", help="6x6 template fc2 yaml (dg0.005, kink present)")
    ap.add_argument(
        "--operator",
        default="",
        help="pre-calibrated long_range_operator.npz; replaces --ref and B/kappa laws",
    )
    ap.add_argument(
        "--short-range-only",
        action="store_true",
        help=(
            "sample base + optional delta model without a harmonic long-range force; "
            "use this to build the TDEP background before a q-space correction"
        ),
    )
    ap.add_argument("--friedel-tel", type=float, default=789)
    ap.add_argument("--temperatures", default="100,300,600")
    ap.add_argument("--cutoff2", type=float, default=6.0)
    ap.add_argument("--friedel-rmax", type=float, default=12.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--equil", type=int, default=1500)
    ap.add_argument("--nsnap", type=int, default=120)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--npoints", type=int, default=201)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-root", default="")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--max-temperature-factor", type=float, default=5.0)
    ap.add_argument("--min-pair-distance", type=float, default=0.8)
    ap.add_argument("--max-force", type=float, default=100.0)
    ap.add_argument("--tag", default="graphene_ft_friedel")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    conditional_delta = bool(a.delta_model_300 or a.delta_model_600)
    if conditional_delta and not (a.delta_model_300 and a.delta_model_600):
        ap.error("--delta-model-300 and --delta-model-600 must be provided together")
    if conditional_delta and a.delta_model:
        ap.error("use either --delta-model or the two conditional endpoint models")
    def log(m): print(m, flush=True)

    temps = [float(x) for x in a.temperatures.split(",")]
    # --- ALL geometry + fc2 from the backbone yaml (atom-order-consistent) ---
    ph0 = fm.load_ph(a.bg)
    fc_bg = ph0.force_constants
    prim = phonopy_to_ase(ph0.unitcell); prim.wrap()
    ideal = phonopy_to_ase(ph0.supercell); ideal.wrap()
    sc_matrix = ph0.supercell_matrix
    log(f"[{a.tag}] {Path(a.bg).name}: prim {len(prim)} atoms, SC {len(ideal)} atoms, cutoff2={a.cutoff2}")
    log(f"[{a.tag}] NOTE prim from yaml unitcell (phonopy round-trip) — if hiphive orbit bug, "
        f"fall back to td_phonon.py build_monolayer prim.")

    base_model = tdc.get_mace_calc(a.model, device=a.device)
    delta_model = None
    delta_model_300 = None
    delta_model_600 = None
    if a.delta_model:
        delta_model = tdc.get_mace_calc(a.delta_model, device=a.device)
    elif conditional_delta:
        delta_model_300 = tdc.get_mace_calc(a.delta_model_300, device=a.device)
        delta_model_600 = tdc.get_mace_calc(a.delta_model_600, device=a.device)

    def short_calculator(temperature):
        if conditional_delta:
            calculator, weights = conditioned_short_calculator(
                base_model, delta_model_300, delta_model_600, temperature
            )
            return calculator, weights
        if delta_model is not None:
            return SumCalculator([base_model, delta_model]), None
        return base_model, None

    corr = None
    if a.short_range_only:
        log(
            f"[{a.tag}] short-range-only MD: frozen short predictor, with no harmonic "
            "long-range force in the trajectory"
        )
    elif a.operator:
        with np.load(a.operator, allow_pickle=False) as payload:
            delta_fc = np.asarray(payload["delta_fc_full"], float)
            operator_cell = np.asarray(payload["cell"], float)
            operator_positions = np.asarray(payload["reference_positions"], float)
        if not np.allclose(operator_cell, np.asarray(ideal.cell), atol=2e-5, rtol=0.0):
            raise ValueError("fixed long-range operator cell does not match MD supercell")
        if not np.allclose(
            operator_positions, np.asarray(ideal.positions), atol=2e-5, rtol=0.0
        ):
            raise ValueError("fixed long-range operator atom order does not match MD supercell")
        corr = FixedHarmonicCorrection(delta_fc)
    else:
        if not a.ref:
            ap.error("provide --operator or --ref")
        fc_ref = fm.load_ph(a.ref).force_constants
        corr = FriedelCorrection(
            ph0,
            fc_bg,
            fc_ref,
            rmin=1.0,
            rmax=a.friedel_rmax,
            B_law=Blaw,
            kappa_law=Klaw,
        )
    if not a.short_range_only:
        log(f"[{a.tag}] FriedelMACE @ T_el={a.friedel_tel}K (short predictor + Friedel long-range). "
            f"Expect kink_K ~ DFT dg0.005 (~13-15 @ 6x6).")

    out = {"temperatures": np.array(temps), "model": np.array(str(a.model)),
           "delta_model": np.array(str(a.delta_model)),
           "delta_model_300": np.array(str(a.delta_model_300)),
           "delta_model_600": np.array(str(a.delta_model_600)),
           "temperature_conditioned_delta": np.array(conditional_delta),
           "operator": np.array(str(a.operator)), "tag": np.array(str(a.tag)),
           "short_range_only": np.array(bool(a.short_range_only)),
           "friedel_tel": np.array(a.friedel_tel), "seed": np.array(a.seed)}
    rows = [
        "T_K,w_gamma_cm,w_k_cm,kink_gamma,kink_k,minfreq_thz,"
        "fit_rmse_meVA,mean_T_K,max_T_K"
    ]
    for T in temps:
        t0 = time.perf_counter()
        short_calc, weights = short_calculator(T)
        if weights is not None:
            out[f"T{int(T)}_delta_weight_300"] = np.array(weights.delta_300)
            out[f"T{int(T)}_delta_weight_600"] = np.array(weights.delta_600)
            log(
                f"[{a.tag}] T={T:.0f} K conditioned short model: "
                f"w300={weights.delta_300:.6f}, w600={weights.delta_600:.6f}"
            )
        calc = (
            short_calc
            if a.short_range_only
            else FriedelMACECalculator(short_calc, ideal, corr, a.friedel_tel)
        )
        log(f"[{a.tag}] T={T:.0f} K: MD sampling ...")
        checkpoint_dir = None
        if a.checkpoint_root:
            checkpoint_root = Path(a.checkpoint_root)
            if not checkpoint_root.is_absolute():
                checkpoint_root = ROOT / checkpoint_root
            checkpoint_dir = checkpoint_root / f"T{int(T)}"
        snaps = tdp.sample_md(
            ideal,
            calc,
            T,
            a.dt,
            a.equil,
            a.nsnap,
            a.stride,
            seed=a.seed,
            log=log,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=a.checkpoint_every,
            max_temperature_factor=a.max_temperature_factor,
            min_pair_distance=a.min_pair_distance,
            max_force=a.max_force,
        )
        instantaneous_temperatures = np.asarray(
            [snapshot.get_temperature() for snapshot in snaps], float
        )
        fc2, rmse, ndof = tdp.effective_fc2(prim, ideal, sc_matrix, snaps, a.cutoff2)
        dist, freq, lp, labs = tdp.band_from_phonopy(ph0, fc2, npoints=a.npoints)
        wG = al.branch_freq_at_label(dist, freq, lp, labs, r"$\Gamma$") * CM
        wK = al.branch_freq_at_label(dist, freq, lp, labs, "K") * CM
        kinks = {k["label"]: k for k in al.high_sym_kinks(dist, freq, lp, labs)}
        kG = kinks[r"$\Gamma$"]["kink_strength"]; kK = kinks["K"]["kink_strength"]
        out[f"T{int(T)}_dist"], out[f"T{int(T)}_freq"] = dist, freq
        out[f"T{int(T)}_fc2"] = fc2
        out[f"T{int(T)}_instantaneous_temperatures_K"] = instantaneous_temperatures
        out[f"T{int(T)}_mean_temperature_K"] = np.array(
            float(np.mean(instantaneous_temperatures))
        )
        out[f"T{int(T)}_max_temperature_K"] = np.array(
            float(np.max(instantaneous_temperatures))
        )
        out["label_positions"], out["labels"] = lp, labs
        rows.append(
            f"{T:.0f},{wG:.2f},{wK:.2f},{kG:.3f},{kK:.3f},"
            f"{freq.min():.3f},{rmse*1000:.2f},"
            f"{np.mean(instantaneous_temperatures):.2f},"
            f"{np.max(instantaneous_temperatures):.2f}"
        )
        log(f"[{a.tag}] T={T:.0f} K: wG={wG:.1f} wK={wK:.1f} kinkG={kG:.1f} kinkK={kK:.1f} "
            f"rmse={rmse*1000:.1f} meV/A ({ndof} dof, {time.perf_counter()-t0:.0f}s)")
    outdir = ROOT / "results" / "td_phonon"; outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / f"td_{a.tag}.npz", **out)
    (outdir / f"td_{a.tag}.csv").write_text("\n".join(rows) + "\n")
    log(f"[{a.tag}] wrote td_{a.tag}.npz + .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
