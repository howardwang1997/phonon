"""V-Q3f -- NbSe2 (E)-channel: the CDW soft-mode frequency vs electronic temperature.

The (L)-channel (lattice-anharmonic) T-evolution of NbSe2's CDW soft mode is given by
TDEP/SSCHA (the MLIP can produce it). The (E)-channel -- the dependence on *electronic*
smearing (Fermi-Dirac degauss = k_B * T_el) -- a ground-state-PES MLIP structurally cannot
produce. As electronic T rises, the Fermi surface smears, the momentum-dependent e-ph
coupling that drives the CDW weakens, and the soft mode should HARDEN (omega^2: negative ->
positive) = the CDW melts. omega^2(degauss) crossing zero estimates the electronic T_CDW.

Method (frozen-phonon along the CDW eigenvector, DFT forces, no DFPT needed):
  load the DFT fc2 (3x3) -> mass-weighted soft eigenvector v (the CDW pattern),
  Cartesian pattern w_i = v_i / sqrt(m_i); displace x0 +- Q0*w; the generalized force
  g(Q) = sum_i F_i . w_i = -dE/dQ; omega^2 = -dg/dQ ~ -(g(+Q0)-g(-Q0))/(2 Q0).
  Do this for each Fermi-Dirac degauss. 2-3 SCF per degauss.

    python scripts/vq3f_nbse2_echannel.py --yaml results/vq3/nbse2_dft_phonopy.yaml \
        --pw "mpirun -np 16 pw.x" --pseudo-dir pseudo \
        --degausses 0.005,0.010,0.015,0.020,0.030,0.040 --q0-disp 0.06 \
        --kpts 6 --ecutwfc 70 --outdir results/vq3f
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

VASP_TO_THZ = 15.633302
THZ_TO_CM = 33.356410
RY_TO_K = 157887.0          # degauss[Ry] -> T_el[K] for Fermi-Dirac

PSEUDOS = {"Nb": "Nb_ONCV_PBE-1.2.upf", "Se": "Se_ONCV_PBE-1.2.upf"}


def soft_mode_pattern(yaml_path):
    """Return (atoms0, w[N,3], v_unit, soft_freq_cm). w_i = v_i/sqrt(m_i), v unit-norm."""
    import phonopy
    from ase import Atoms
    ph = phonopy.load(yaml_path, is_compact_fc=False)
    sc = ph.supercell
    atoms0 = Atoms(symbols=list(sc.symbols), scaled_positions=sc.scaled_positions,
                   cell=np.array(sc.cell), pbc=[True, True, True])
    masses = atoms0.get_masses()
    N = len(atoms0)
    fc = np.asarray(ph.force_constants)
    if fc.shape[0] != N:
        from phonopy.harmonic.force_constants import compact_fc_to_full_fc
        fc = np.asarray(compact_fc_to_full_fc(ph.primitive, fc))
    D = fc.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    D = 0.5 * (D + D.T)
    msqrt = np.repeat(np.sqrt(masses), 3)
    Dw = D / np.outer(msqrt, msqrt)
    w2, V = np.linalg.eigh(Dw)
    k = int(np.argmin(w2))                          # softest (CDW) mode
    v = V[:, k]                                     # mass-weighted, unit-norm
    w = (v / msqrt).reshape(N, 3)                   # Cartesian displacement pattern
    soft_cm = float(np.sign(w2[k]) * np.sqrt(abs(w2[k])) * VASP_TO_THZ * THZ_TO_CM)
    return atoms0, w, v, soft_cm


def make_espresso(pw, pseudo_dir, ecutwfc, ecutrho, kpts, degauss, smearing, directory):
    from ase.calculators.espresso import Espresso, EspressoProfile
    profile = EspressoProfile(command=pw, pseudo_dir=str(pseudo_dir))
    input_data = {
        "control": {"calculation": "scf", "tprnfor": True, "tstress": False,
                    "disk_io": "none", "verbosity": "low"},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                   "smearing": smearing, "degauss": degauss},
        "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.3, "electron_maxstep": 250,
                      "diago_david_ndim": 4, "startingwfc": "atomic+random"},
    }
    return Espresso(profile=profile, pseudopotentials=PSEUDOS, input_data=input_data,
                    kpts=(kpts, kpts, 1), directory=Path(directory))


def gforce(atoms0, w, Q, x0, calc_factory, tag, workdir):
    """Generalized force g(Q) = sum_i F_i . w_i at displacement x0 + Q*w."""
    import shutil
    at = atoms0.copy(); at.set_positions(x0 + Q * w)
    d = Path(workdir) / tag; d.mkdir(parents=True, exist_ok=True)
    at.calc = calc_factory(d)
    e = float(at.get_potential_energy())
    F = np.asarray(at.get_forces())
    g = float((F * w).sum())
    shutil.rmtree(d, ignore_errors=True)
    return e, g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="results/vq3/nbse2_dft_phonopy.yaml")
    ap.add_argument("--pw", required=True, help='e.g. "mpirun -np 16 pw.x" or /root/gpupw.sh')
    ap.add_argument("--pseudo-dir", default="pseudo")
    ap.add_argument("--degausses", default="0.005,0.010,0.015,0.020,0.030,0.040",
                    help="Fermi-Dirac degauss (Ry); T_el = degauss*157887 K")
    ap.add_argument("--smearing", default="fermi-dirac")
    ap.add_argument("--q0-disp", type=float, default=0.06, help="max atomic disp (A) at +-Q0")
    ap.add_argument("--kpts", type=int, default=6)
    ap.add_argument("--ecutwfc", type=float, default=70.0)
    ap.add_argument("--ecutrho", type=float, default=280.0)
    ap.add_argument("--workdir", default="/data/vq3f_work")
    ap.add_argument("--outdir", default="results/vq3f")
    a = ap.parse_args()

    yaml_path = a.yaml
    atoms0, w, v, soft_cm = soft_mode_pattern(yaml_path)
    x0 = atoms0.get_positions()
    Q0 = a.q0_disp / np.abs(w).max()                # so max atomic disp = q0-disp
    degausses = [float(x) for x in a.degausses.split(",") if x.strip()]
    pseudo_dir = Path(a.pseudo_dir)
    print(f"[vq3f] {len(atoms0)} atoms; bare-DFT soft mode {soft_cm:.1f} cm^-1; "
          f"Q0={Q0:.4f} (max disp {a.q0_disp} A); smearing={a.smearing}", flush=True)
    print(f"[vq3f] degauss scan (Ry -> T_el K): "
          + ", ".join(f"{dg}->{dg*RY_TO_K:.0f}" for dg in degausses), flush=True)

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for dg in degausses:
        def factory(d, dg=dg):
            return make_espresso(a.pw, pseudo_dir, a.ecutwfc, a.ecutrho, a.kpts,
                                 dg, a.smearing, d)
        ep, gp = gforce(atoms0, w, +Q0, x0, factory, f"dg{dg}_p", a.workdir)
        em, gm = gforce(atoms0, w, -Q0, x0, factory, f"dg{dg}_m", a.workdir)
        # omega^2 = -dg/dQ ~ -(g(+Q0)-g(-Q0))/(2 Q0)   [eV/(A^2 amu)]
        w2 = -(gp - gm) / (2 * Q0)
        freq_cm = float(np.sign(w2) * np.sqrt(abs(w2)) * VASP_TO_THZ * THZ_TO_CM)
        Tel = dg * RY_TO_K
        rows.append((dg, Tel, w2, freq_cm))
        print(f"[vq3f] degauss={dg} (T_el={Tel:.0f} K): "
              f"soft-mode omega^2={w2:+.4f} -> {freq_cm:+.1f} cm^-1  "
              f"({'SOFT' if freq_cm < 0 else 'stable'}) [{time.perf_counter()-t0:.0f}s]",
              flush=True)
        # incremental save
        import csv
        with open(outdir / "nbse2_echannel.csv", "w", newline="") as fh:
            wtr = csv.writer(fh); wtr.writerow(["degauss_Ry", "T_el_K", "omega2_eVA2amu", "freq_cm"])
            wtr.writerows(rows)

    arr = np.array(rows)
    np.savez(outdir / "nbse2_echannel.npz", degauss=arr[:, 0], T_el=arr[:, 1],
             omega2=arr[:, 2], freq_cm=arr[:, 3], soft_cm_bare=soft_cm)
    # zero-crossing estimate of electronic T_CDW
    fc = arr[:, 3]
    cross = ""
    neg = np.where(fc < 0)[0]
    if len(neg) and neg[-1] + 1 < len(fc):
        i = neg[-1]
        Ta, Tb, fa, fb = arr[i, 1], arr[i + 1, 1], fc[i], fc[i + 1]
        Tc = Ta + (Tb - Ta) * (0 - fa) / (fb - fa)
        cross = f"  electronic T_CDW ~ {Tc:.0f} K (omega^2 zero-crossing)"
    print(f"[vq3f] saved -> {outdir}/nbse2_echannel.csv/.npz{cross}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
