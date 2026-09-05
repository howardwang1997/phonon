"""DFT-MD-TDEP for graphene (L) baseline: ASE Langevin MD with Espresso pw.x ->
effective fc2(T) -> band structure. Gives the DFT (L) reference to compare the MLIP (L).

Reuses td_phonon.sample_md / effective_fc2 / band_from_phonopy (calc-agnostic).
Run on Box A/B (V100 QE):
  python scripts/smearing_kink/dft_md_tdep.py \
    --phonopy results/vq_kink6/graphene_sc6_dg0.002_phonopy.yaml \
    --pw /root/gpupw.sh --pseudo-dir /root/phonon/pseudo --tag graphene_dft_tdep
"""
from __future__ import annotations
import argparse, os, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import td_phonon as tdp
from ase.io import read
from phonopy import Phonopy
from phonon_accel.phonons import ase_to_phonopy, phonopy_to_ase
CM = 33.356


def make_espresso(pw, pseudo_dir, ecutwfc, ecutrho, kpts, degauss, directory, numbers=None):
    from ase.data import chemical_symbols
    from ase.calculators.espresso import Espresso, EspressoProfile
    if numbers is not None:
        els = sorted(set(chemical_symbols[z] for z in numbers))
        PSEUDO = {el: f"{el}_ONCV_PBE-1.2.upf" for el in els}
    else:
        PSEUDO = {"C": "C_ONCV_PBE-1.2.upf"}
    profile = EspressoProfile(command=pw, pseudo_dir=str(pseudo_dir))
    inp = {"control": {"calculation": "scf", "tprnfor": True, "tstress": False,
                       "prefix": "md", "pseudo_dir": str(pseudo_dir)},
           "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                      "smearing": "cold", "degauss": degauss, "ibrav": 0},
           "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.3, "electron_maxstep": 200}}
    return Espresso(profile=profile, pseudopotentials=PSEUDO, input_data=inp,
                    kpts=kpts, directory=str(directory))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phonopy", required=True, help="graphene fc2 yaml (primitive/unit cell)")
    ap.add_argument("--supercell", default="6,6,1")   # K=(1/3,1/3) on-grid; matches (E) sweep + td_phonon
    ap.add_argument("--pw", required=True)
    ap.add_argument("--pseudo-dir", required=True)
    ap.add_argument("--ecutwfc", type=float, default=60.0)
    ap.add_argument("--ecutrho", type=float, default=480.0)
    ap.add_argument("--kpts", default="4,4,1")         # on 6x6 cell = 24x24 primitive equiv (dense enough for Kohn)
    ap.add_argument("--degauss", type=float, default=0.005)  # FIXED ref smearing = (E)-channel low anchor; isolates (L)
    ap.add_argument("--temperatures", default="100,300,600")
    ap.add_argument("--nsnap", type=int, default=60)   # was 30 -> rmse_fc2 67; 60 targets <10 meV/A
    ap.add_argument("--equil", type=int, default=500)  # was 50; proper thermalization
    ap.add_argument("--stride", type=int, default=20)  # was 10; decorrelation
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--cutoff2", type=float, default=6.0)  # match td_phonon.py
    ap.add_argument("--workroot", default="/data/gr_dftmd")
    ap.add_argument("--tag", default="graphene_dft_tdep")
    ap.add_argument("--mlip-model", default=None, help="if set, use MACE MLIP (not Espresso) for MD")
    ap.add_argument("--checkpoint-every", type=int, default=1,
                    help="atomically checkpoint every N MD steps (default: every costly DFT step)")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--initial-structure", default=None,
                    help="seed positions from a legacy interrupted ASE/QE input if no checkpoint exists")
    ap.add_argument("--provenance-note", default="")
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
    out = {"temperatures": np.array(temps), "tag": np.array(a.tag),
           "provenance_note": np.array(a.provenance_note),
           "initial_structure": np.array(a.initial_structure or ""),
           "checkpoint_every": int(a.checkpoint_every)}
    rows = ["T_K,minfreq_cm,rmse_fc2_meVA"]
    failures = []
    for T in temps:
        t0 = time.perf_counter()
        work = Path(a.workroot) / f"T{int(T)}"
        work.mkdir(parents=True, exist_ok=True)
        md_ideal = ideal.copy()
        checkpoint_dir = work / "md_checkpoint"
        if a.initial_structure and not (checkpoint_dir / "state.npz").is_file():
            seed_atoms = read(a.initial_structure, format="espresso-in")
            if len(seed_atoms) != len(md_ideal) or not np.array_equal(seed_atoms.numbers, md_ideal.numbers):
                raise ValueError(
                    f"initial structure does not match the phonopy supercell: "
                    f"{len(seed_atoms)} vs {len(md_ideal)} atoms"
                )
            if not np.allclose(seed_atoms.cell, md_ideal.cell, rtol=0.0, atol=1e-5):
                raise ValueError("initial structure cell differs from the phonopy MD cell")
            md_ideal.set_positions(seed_atoms.positions)
            print(f"[{a.tag}] T={T:.0f} K: seeded positions from {a.initial_structure}", flush=True)
        if a.mlip_model:
            from mace.calculators import MACECalculator
            calc = MACECalculator(model_paths=a.mlip_model, device="cuda", default_dtype="float32")
        else:
            calc = make_espresso(a.pw, a.pseudo_dir, a.ecutwfc, a.ecutrho, kpts, a.degauss, work, numbers=ideal.numbers)
        print(f"[{a.tag}] T={T:.0f} K: DFT-MD (n_snap={a.nsnap}, equil={a.equil}, stride={a.stride})...", flush=True)
        snaps = tdp.sample_md(
            md_ideal, calc, T, a.dt, a.equil, a.nsnap, a.stride,
            seed=int(T), log=print, checkpoint_dir=checkpoint_dir,
            resume=a.resume, checkpoint_every=a.checkpoint_every,
        )
        # save snapshots so the fc2 fit can be re-tuned without re-running the (slow) DFT-MD
        try:
            energies = np.array([float(s.get_potential_energy()) for s in snaps])
        except Exception:
            energies = np.zeros(len(snaps))
        tdp._atomic_savez(
            work / f"snaps_T{int(T)}.npz",
            positions=np.array([s.positions for s in snaps]),
            forces=np.array([s.get_forces() for s in snaps]), energies=energies,
            numbers=ideal.numbers, cell=ideal.cell,
            source_checkpoint=np.array(str(checkpoint_dir)),
            provenance_note=np.array(a.provenance_note),
        )
        try:
            fc2, rmse2, _ = tdp.effective_fc2(prim, ideal, scm, snaps, a.cutoff2)
            dist, freq, lp, labs = tdp.band_from_phonopy(ph0, fc2, npoints=60)
        except Exception as e:
            print(f"[{a.tag}] T={T:.0f} K: fc2 fit FAILED ({type(e).__name__}: {e}); skipping", flush=True)
            failures.append(f"T={T:g}: {type(e).__name__}: {e}")
            continue
        out[f"T{int(T)}_dist"] = dist; out[f"T{int(T)}_freq"] = freq
        out["label_positions"] = lp; out["labels"] = labs
        minf = float(freq.min()) * CM
        rows.append(f"{T:.0f},{minf:.2f},{rmse2*1000:.1f}")
        print(f"[{a.tag}] T={T:.0f} K: min={minf:.1f} cm^-1, rmse_fc2={rmse2*1000:.1f} meV/A; "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
    outdir = ROOT / "results" / "td_phonon"; outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{a.tag}.csv"
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    with csv_tmp.open("w") as handle:
        handle.write("\n".join(rows) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(csv_tmp, csv_path)
    tdp._atomic_savez(outdir / f"{a.tag}.npz", **out)
    print("wrote", outdir / f"{a.tag}.npz", flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))
    if len(rows) != len(temps) + 1:
        raise RuntimeError(f"only {len(rows)-1}/{len(temps)} temperatures completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
