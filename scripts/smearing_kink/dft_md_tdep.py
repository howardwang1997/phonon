"""DFT-MD-TDEP for graphene (L) baseline: ASE Langevin MD with Espresso pw.x ->
effective fc2(T) -> band structure. Gives the DFT (L) reference to compare the MLIP (L).

Reuses td_phonon.sample_md / effective_fc2 / band_from_phonopy (calc-agnostic).
Run on Box A/B (V100 QE):
  python scripts/smearing_kink/dft_md_tdep.py \
    --phonopy results/vq_kink6/graphene_sc6_dg0.002_phonopy.yaml \
    --pw /root/gpupw.sh --pseudo-dir /root/phonon/pseudo --tag graphene_dft_tdep
"""
from __future__ import annotations
import sys, time, warnings, argparse
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import td_phonon as tdp
from phonopy import Phonopy
from phonon_accel.phonons import ase_to_phonopy, phonopy_to_ase
CM = 33.356


def make_espresso(pw, pseudo_dir, ecutwfc, ecutrho, kpts, degauss, directory):
    from ase.calculators.espresso import Espresso, EspressoProfile
    PSEUDO = {"C": "C_ONCV_PBE-1.2.upf"}
    profile = EspressoProfile(command=pw, pseudo_dir=str(pseudo_dir))
    inp = {"control": {"calculation": "scf", "tprnfor": True, "tstress": False,
                       "prefix": "graphene", "pseudo_dir": str(pseudo_dir)},
           "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                      "smearing": "cold", "degauss": degauss, "ibrav": 0},
           "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.3, "electron_maxstep": 200}}
    return Espresso(profile=profile, pseudopotentials=PSEUDO, input_data=inp,
                    kpts=kpts, directory=str(directory))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phonopy", required=True, help="graphene fc2 yaml (primitive/unit cell)")
    ap.add_argument("--supercell", default="4,4,1")
    ap.add_argument("--pw", required=True)
    ap.add_argument("--pseudo-dir", required=True)
    ap.add_argument("--ecutwfc", type=float, default=60.0)
    ap.add_argument("--ecutrho", type=float, default=480.0)
    ap.add_argument("--kpts", default="4,4,1")
    ap.add_argument("--degauss", type=float, default=0.01)
    ap.add_argument("--temperatures", default="300,1000")
    ap.add_argument("--nsnap", type=int, default=30)
    ap.add_argument("--equil", type=int, default=50)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--cutoff2", type=float, default=4.5)
    ap.add_argument("--workroot", default="/data/gr_dftmd")
    ap.add_argument("--tag", default="graphene_dft_tdep")
    a = ap.parse_args()

    import phonopy
    ph_y = phonopy.load(a.phonopy, is_compact_fc=False)
    prim = phonopy_to_ase(ph_y.unitcell); prim.wrap()
    scm = np.diag([int(x) for x in a.supercell.split(",")])
    ph0 = Phonopy(ase_to_phonopy(prim), supercell_matrix=scm, primitive_matrix=np.eye(3))
    ideal = phonopy_to_ase(ph0.supercell); ideal.wrap()
    print(f"[{a.tag}] graphene primitive {len(prim)} atoms; MD supercell {len(ideal)} atoms; "
          f"ecut={a.ecutwfc} kpts={a.kpts} dg={a.degauss}", flush=True)

    kpts = [int(x) for x in a.kpts.split(",")]
    temps = [float(x) for x in a.temperatures.split(",")]
    out = {"temperatures": np.array(temps), "tag": a.tag}
    rows = ["T_K,minfreq_cm,rmse_fc2_meVA"]
    for T in temps:
        t0 = time.perf_counter()
        work = Path(a.workroot) / f"T{int(T)}"
        work.mkdir(parents=True, exist_ok=True)
        calc = make_espresso(a.pw, a.pseudo_dir, a.ecutwfc, a.ecutrho, kpts, a.degauss, work)
        print(f"[{a.tag}] T={T:.0f} K: DFT-MD (n_snap={a.nsnap}, equil={a.equil}, stride={a.stride})...", flush=True)
        snaps = tdp.sample_md(ideal, calc, T, a.dt, a.equil, a.nsnap, a.stride, seed=int(T), log=print)
        # save snapshots so the fc2 fit can be re-tuned without re-running the (slow) DFT-MD
        np.savez(work / f"snaps_T{int(T)}.npz",
                 positions=np.array([s.positions for s in snaps]),
                 forces=np.array([s.get_forces() for s in snaps]),
                 numbers=ideal.numbers, cell=ideal.cell)
        try:
            fc2, rmse2, _ = tdp.effective_fc2(prim, ideal, scm, snaps, a.cutoff2)
            dist, freq, lp, labs = tdp.band_from_phonopy(ph0, fc2, npoints=60)
        except Exception as e:
            print(f"[{a.tag}] T={T:.0f} K: fc2 fit FAILED ({type(e).__name__}: {e}); skipping", flush=True)
            continue
        out[f"T{int(T)}_dist"] = dist; out[f"T{int(T)}_freq"] = freq
        out["label_positions"] = lp; out["labels"] = labs
        minf = float(freq.min()) * CM
        rows.append(f"{T:.0f},{minf:.2f},{rmse2*1000:.1f}")
        print(f"[{a.tag}] T={T:.0f} K: min={minf:.1f} cm^-1, rmse_fc2={rmse2*1000:.1f} meV/A; "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
    outdir = ROOT / "results" / "td_phonon"; outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.tag}.csv").write_text("\n".join(rows) + "\n")
    np.savez(outdir / f"{a.tag}.npz", **out)
    print("wrote", outdir / f"{a.tag}.npz", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
