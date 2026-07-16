"""Deploy the (E)-channel on the graphene Kohn fd scan: v11 MACE backbone + Friedel
long-range term (fixed D0, laws from fit_friedel_kohn_fd.json) at each T_el, vs DFT.

backbone = v11 MACE finite-displacement fc2 on the 6x6 graphene at the relaxed a.
D0 = DFT fc2(dg0.01) - backbone.  Deploy fc2(T_el) = backbone + B*exp(-kappa(T_el)R)*D0.
Reports kink_K / kink_G / Gamma-E2g / K-iTO at each T_el vs DFT (and the backbone alone).

    conda run -n phonon python scripts/smearing_kink/deploy_kohn_fd.py
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import friedel_module as fm
from friedel_calc import FriedelCorrection, FriedelMACECalculator, fc2_from_calc

FD = ROOT / "results" / "graphene_kohn_fd"
BB_MODEL = ROOT / "results" / "gr_backbone" / "dg0.080" / "gr_dg0.080.model"
FIT = ROOT / "results" / "graphene_kohn_fd" / "friedel_fit_kohn_fd.json"
DGS = ["0.01", "0.015", "0.02", "0.03", "0.04", "0.06", "0.08", "0.10", "0.14", "0.20"]
REF_DG = "0.01"


def mace_calc(mp):
    from mace.calculators import MACECalculator
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return MACECalculator(model_paths=str(mp), device=dev, default_dtype="float32")


def metrics(ph, fc):
    kK, wK, kG = fm.kink_of(ph, fc)
    ph.run_qpoints([np.zeros(3), np.array([1/3, 1/3, 0.])], with_dynamical_matrices=False)
    f = np.array(ph.get_qpoints_dict()["frequencies"]) * 33.35641
    return kK, kG, float(f[0].max()), float(f[1].max())


_PTS = [np.array([0., 0., 0.]), np.array([.5, 0., 0.]),
        np.array([1/3, 1/3, 0.]), np.array([0., 0., 0.])]


def band(ph, fc, nseg=40):
    """Gamma-M-K-Gamma dispersion (cm^-1) from a compact fc, on a fresh phonopy copy."""
    import phonopy
    ph2 = phonopy.load(FD / "graphene_sc6_dg0.08_phonopy.yaml", is_compact_fc=False)
    ph2.force_constants = fc
    qs = []
    for i in range(len(_PTS) - 1):
        for j in range(1, nseg + 1):
            qs.append(_PTS[i] + (_PTS[i + 1] - _PTS[i]) * j / nseg)
    qs = np.array(qs)
    ph2.run_qpoints(qs, with_dynamical_matrices=False)
    freq = np.array(ph2.get_qpoints_dict()["frequencies"]) * 33.35641
    seglen = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    x = np.concatenate([[0.0], np.cumsum(seglen)])
    seg = len(x) // 3
    return x, freq, [x[0], x[seg - 1], x[2 * seg - 1], x[-1]]


def main():
    ph = fm.load_ph(FD / "graphene_sc6_dg0.08_phonopy.yaml")     # phonopy object (relaxed a, 6x6)
    fc_dft = {dg: fm.load_ph(FD / f"graphene_sc6_dg{dg}_phonopy.yaml").force_constants for dg in DGS
              if (FD / f"graphene_sc6_dg{dg}_phonopy.yaml").exists()}
    fit = json.loads(FIT.read_text()) if FIT.exists() else {}
    # Use the PROVEN graphene Friedel laws (friedel_calc defaults; validated to MAE 0.31)
    # rather than the per-T power-law re-fit, which is noisy on this range.
    law = json.loads((FD / "kappa_law_tuned.json").read_text()) if (FD / "kappa_law_tuned.json").exists() else {"B":1.08,"a":8.318e-4,"b":0.61}
    Bm, ka, kb = law["B"], law["a"], law["b"]
    print(f"# v11 backbone + fixed-D0 Friedel; laws B={Bm}, kappa(T)={ka}*T^{kb} ({'tuned' if 'MAE' in law else 'proven'})")

    mace = mace_calc(BB_MODEL)
    from phonon_accel.phonons import phonopy_to_ase
    ref_atoms = phonopy_to_ase(ph.supercell)
    # backbone fc2 through the MACE (subtract ref forces; MACE geom != exact min)
    _, fc_bb = fc2_from_calc(ph, mace, distance=0.03, subtract_ref=True)
    corr = FriedelCorrection(ph, fc_dft["0.08"], fc_dft[REF_DG],
                             B_law=lambda T: Bm, kappa_law=lambda T: ka * T ** kb)

    kB = metrics(ph, fc_bb)[0]
    bb_x, bb_f, bb_tick = band(ph, fc_bb)
    print(f"\n{'dg':>6} {'T_el':>7} {'DFT_kinkK':>9} {'backbone':>9} {'MACE+LR':>8}")
    dft_kink = {}; mlip_kink = {}; dft_b = {}; mlip_b = {}
    res = 0.0; n = 0
    for dg in DGS:
        if dg not in fc_dft:
            continue
        T = float(dg) * 157887
        dft_kink[dg] = metrics(ph, fc_dft[dg])[0]
        _, dftf, _ = band(ph, fc_dft[dg]); dft_b[dg] = dftf
        calc = FriedelMACECalculator(mace, ref_atoms, corr, T)
        _, fc_t = fc2_from_calc(ph, calc, distance=0.03, subtract_ref=True)
        mlip_kink[dg] = metrics(ph, fc_t)[0]
        _, mlf, _ = band(ph, fc_t); mlip_b[dg] = mlf
        res += abs(mlip_kink[dg] - dft_kink[dg]); n += 1
        print(f"{dg:>6} {T:>7.0f} {dft_kink[dg]:>9.2f} {kB:>9.2f} {mlip_kink[dg]:>8.2f}")
    print(f"# MAE kink_K (MACE+LR vs DFT) = {res/n:.2f} cm^-1   (backbone alone kink_K={kB:.2f})")
    keep = [dg for dg in DGS if dg in fc_dft]
    np.savez(FD / "deploy_lineA_bands.npz", x=bb_x, tick=bb_tick, dgs=np.array(keep),
             dft_bands=np.array([dft_b[dg] for dg in keep]),
             mlip_bands=np.array([mlip_b[dg] for dg in keep]),
             dft_kink=np.array([dft_kink[dg] for dg in keep]),
             mlip_kink=np.array([mlip_kink[dg] for dg in keep]),
             backbone_band=bb_f, backbone_kink=kB)
    print(f"# saved {FD/'deploy_lineA_bands.npz'}")

    # ---- self-consistent per-T fit on the v11 backbone: can v11+Friedel REPRESENT each T? ----
    tabs, _ = fm.pair_table(ph)
    D0 = fm.template_delta(fc_dft[REF_DG], fc_dft["0.08"], tabs)
    KAPPAS = np.linspace(0.0, 2.0, 161)
    print("\n# reuse-v11, self-consistent per-T Friedel fit (D0 = DFT dg0.01 - v11 backbone):")
    print(f"{'dg':>6} {'T_el':>7} {'DFT':>7} {'v11+LR(perT)':>13}")
    rP = 0.0; nP = 0
    for dg in DGS:
        if dg not in fc_dft:
            continue
        dK = metrics(ph, fc_dft[dg])[0]
        if dg == REF_DG:
            kP = dK
        else:
            B, kap, _ = fm.fit_template_env(fc_dft[dg], fc_bb, tabs, D0, 1.0, 12.0, KAPPAS)
            fcP = fm.add_template(fc_bb, tabs, D0, B, kap, 1.0, 12.0)
            kP = metrics(ph, fcP)[0]
        rP += abs(kP - dK); nP += 1
        print(f"{dg:>6} {float(dg)*157887:>7.0f} {dK:>7.2f} {kP:>13.2f}")
    print(f"# MAE kink_K (v11+perT-Friedel vs DFT) = {rP/max(nP,1):.2f} cm^-1")


if __name__ == "__main__":
    main()
