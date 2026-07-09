"""Deploy MLIP+long-range (FriedelMACE) -> full band structure, compare to DFT.
Shows whether the method (MACE backbone + T_el-conditioned Friedel long-range term)
reproduces the DFT phonon spectrum, including the Kohn anomaly.

For each T_el: deployed spectrum (distilled backbone + Friedel(T_el)) vs DFT spectrum (target fc2).
Run on 2060:
  python scripts/smearing_kink/deploy_band_compare.py \
    --mat NbSe2 --backbone-yaml results/v100/fc2_nbse2_3x3_0.020/NbSe2_phonopy.yaml \
    --template-yaml results/v100/fc2_nbse2_3x3_0.005/NbSe2_phonopy.yaml \
    --model results/fam_backbone/NbSe2/ft_NbSe2.model \
    --dft-yaml results/v100/fc2_nbse2_3x3_0.005/NbSe2_phonopy.yaml,results/v100/fc2_nbse2_3x3_0.020/NbSe2_phonopy.yaml \
    --tel 789,3158 --tag deploy_vs_dft --device cuda
"""
from __future__ import annotations
import sys, csv, warnings, argparse
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))
import friedel_module as fm
from friedel_calc import FriedelCorrection, FriedelMACECalculator, fc2_from_calc
from phonon_accel.phonons import phonopy_to_ase

SK = ROOT / "results" / "smearing_kink"
POINTS = [np.array([0., 0., 0.]), np.array([.5, 0., 0.]),
          np.array([1/3, 1/3, 0.]), np.array([0., 0., 0.])]


def qpath(nseg=40):
    qs = []
    for i in range(len(POINTS) - 1):
        q0, q1 = POINTS[i], POINTS[i + 1]
        for j in range(1, nseg + 1):
            qs.append(q0 + (q1 - q0) * j / nseg)
    return np.array(qs)


def bands(ph, fc2, qs):
    ph.force_constants = fc2
    ph.run_qpoints(qs, with_dynamical_matrices=False)
    return np.array(ph.get_qpoints_dict()["frequencies"]) * 33.356  # THz -> cm^-1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--backbone-yaml", required=True, help="dg0.020 fc2 yaml (backbone ref)")
    ap.add_argument("--template-yaml", required=True, help="dg0.005 fc2 yaml (Friedel template)")
    ap.add_argument("--model", required=True, help="distilled MACE backbone")
    ap.add_argument("--dft-yaml", required=True, help="comma-sep DFT target fc2 yamls (one per T_el)")
    ap.add_argument("--tel", default="789")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="deploy_vs_dft")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(SK / "friedel_family.csv")) if r["material"] == a.mat]
    Tb = np.array([float(r["T_el"]) for r in rows]); Bb = np.array([float(r["B"]) for r in rows])
    Kb = np.array([float(r["kappa"]) for r in rows]); o = np.argsort(Tb)
    Tb, Bb, Kb = Tb[o], Bb[o], Kb[o]
    Blaw = lambda T: float(np.interp(T, Tb, Bb)); Klaw = lambda T: float(np.interp(T, Tb, Kb))

    ph = fm.load_ph(a.backbone_yaml); fc_bg = ph.force_constants
    fc_sh = fm.load_ph(a.template_yaml).force_constants
    corr = FriedelCorrection(ph, fc_bg, fc_sh, rmin=1.0, rmax=14.0, B_law=Blaw, kappa_law=Klaw)
    ref = phonopy_to_ase(ph.supercell)
    from mace.calculators import MACECalculator
    base = MACECalculator(model_paths=str(a.model), device=a.device, default_dtype="float32")
    qs = qpath(40)
    out = {"qs": qs, "tel": np.array([float(x) for x in a.tel.split(",")])}

    # smearing-blind backbone (reference: what MLIP alone gives, no Friedel)
    _, fc_bb = fc2_from_calc(ph, base, subtract_ref=True)
    out["backbone"] = bands(ph, fc_bb, qs)
    print(f"backbone (smearing-blind) min = {out['backbone'].min():.1f} cm^-1")

    dft_yamls = a.dft_yaml.split(",")
    for i, T in enumerate(out["tel"]):
        T = float(T)
        calc = FriedelMACECalculator(base, ref, corr, T)
        _, fc_dep = fc2_from_calc(ph, calc, subtract_ref=True)
        out[f"mlip_T{int(T)}"] = bands(ph, fc_dep, qs)
        ph_d = fm.load_ph(dft_yamls[i]); out[f"dft_T{int(T)}"] = bands(ph_d, ph_d.force_constants, qs)
        mae = np.mean(np.abs(out[f"mlip_T{int(T)}"] - out[f"dft_T{int(T)}"]))
        print(f"T_el={T:.0f}: MLIP min={out[f'mlip_T{int(T)}'].min():.1f}  DFT min={out[f'dft_T{int(T)}'].min():.1f}  "
              f"cm^-1  full-band MAE={mae:.2f} cm^-1")
    np.savez(SK / f"{a.tag}_{a.mat}.npz", **out)
    print("wrote", SK / f"{a.tag}_{a.mat}.npz")


if __name__ == "__main__":
    raise SystemExit(main())
