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
from friedel_calc import FriedelCorrection, FriedelMACECalculator
from phonon_accel.phonons import phonopy_to_ase
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
    ap.add_argument("--bg", required=True, help="6x6 backbone fc2 yaml (dg0.080, kink-free)")
    ap.add_argument("--ref", required=True, help="6x6 template fc2 yaml (dg0.005, kink present)")
    ap.add_argument("--friedel-tel", type=float, default=789)
    ap.add_argument("--temperatures", default="100,300,600")
    ap.add_argument("--cutoff2", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--equil", type=int, default=1500)
    ap.add_argument("--nsnap", type=int, default=120)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--npoints", type=int, default=201)
    ap.add_argument("--tag", default="graphene_ft_friedel")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    def log(m): print(m, flush=True)

    temps = [float(x) for x in a.temperatures.split(",")]
    # --- ALL geometry + fc2 from the backbone yaml (atom-order-consistent) ---
    ph0 = fm.load_ph(a.bg)
    fc_bg = ph0.force_constants
    fc_ref = fm.load_ph(a.ref).force_constants
    corr = FriedelCorrection(ph0, fc_bg, fc_ref, rmin=1.0, rmax=9.8, B_law=Blaw, kappa_law=Klaw)
    prim = phonopy_to_ase(ph0.unitcell); prim.wrap()
    ideal = phonopy_to_ase(ph0.supercell); ideal.wrap()
    sc_matrix = ph0.supercell_matrix
    log(f"[{a.tag}] {Path(a.bg).name}: prim {len(prim)} atoms, SC {len(ideal)} atoms, cutoff2={a.cutoff2}")
    log(f"[{a.tag}] NOTE prim from yaml unitcell (phonopy round-trip) — if hiphive orbit bug, "
        f"fall back to td_phonon.py build_monolayer prim.")

    base = tdc.get_mace_calc(a.model, device=a.device)
    calc = FriedelMACECalculator(base, ideal, corr, a.friedel_tel)
    log(f"[{a.tag}] FriedelMACE @ T_el={a.friedel_tel}K (MACE backbone + Friedel long-range). "
        f"Expect kink_K ~ DFT dg0.005 (~13-15 @ 6x6).")

    out = {"temperatures": np.array(temps), "model": np.array(str(a.model)),
           "tag": np.array(str(a.tag)), "friedel_tel": np.array(a.friedel_tel)}
    rows = ["T_K,w_gamma_cm,w_k_cm,kink_gamma,kink_k,minfreq_thz,fit_rmse_meVA"]
    for T in temps:
        t0 = time.perf_counter()
        log(f"[{a.tag}] T={T:.0f} K: MD sampling ...")
        snaps = tdp.sample_md(ideal, calc, T, a.dt, a.equil, a.nsnap, a.stride, log=log)
        fc2, rmse, ndof = tdp.effective_fc2(prim, ideal, sc_matrix, snaps, a.cutoff2)
        dist, freq, lp, labs = tdp.band_from_phonopy(ph0, fc2, npoints=a.npoints)
        wG = al.branch_freq_at_label(dist, freq, lp, labs, r"$\Gamma$") * CM
        wK = al.branch_freq_at_label(dist, freq, lp, labs, "K") * CM
        kinks = {k["label"]: k for k in al.high_sym_kinks(dist, freq, lp, labs)}
        kG = kinks[r"$\Gamma$"]["kink_strength"]; kK = kinks["K"]["kink_strength"]
        out[f"T{int(T)}_dist"], out[f"T{int(T)}_freq"] = dist, freq
        out["label_positions"], out["labels"] = lp, labs
        rows.append(f"{T:.0f},{wG:.2f},{wK:.2f},{kG:.3f},{kK:.3f},{freq.min():.3f},{rmse*1000:.2f}")
        log(f"[{a.tag}] T={T:.0f} K: wG={wG:.1f} wK={wK:.1f} kinkG={kG:.1f} kinkK={kK:.1f} "
            f"rmse={rmse*1000:.1f} meV/A ({ndof} dof, {time.perf_counter()-t0:.0f}s)")
    outdir = ROOT / "results" / "td_phonon"; outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / f"td_{a.tag}.npz", **out)
    (outdir / f"td_{a.tag}.csv").write_text("\n".join(rows) + "\n")
    log(f"[{a.tag}] wrote td_{a.tag}.npz + .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
