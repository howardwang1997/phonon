"""P2-b: Fermi-surface nesting function xi(q) for ANY TMD family member (generalizes
vq3d_nbse2_nesting.py). Builds the monolayer via ase.build.mx2 (2H|1T), runs a dense
DFT grid, and computes xi(q) = autocorrelation of the FS density map via FFT.

xi(q) peaking at the material's CDW wavevector => nesting-driven; else momentum-
dependent e-ph coupling. Material params (formula, polytype, a, thickness, M, X) are
read from configs/v100_campaign.yaml by --name.

    python scripts/v100/chi_q_family.py --name 2H-TaS2 --pw /root/gpupw.sh --nproc 1 \
        --pseudo-dir /root/phonon/pseudo --nk 36 --sigma 0.12
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
PSEUDO = {e: f"{e}_ONCV_PBE-1.2.upf" for e in ("Nb", "Ta", "Ti", "V", "S", "Se", "Mo", "W")}


def load_mat(name):
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs/v100_campaign.yaml"))
    for m in cfg["materials"]:
        if m["name"] == name:
            return m
    raise SystemExit(f"{name} not in config")


def espresso(pw, mpirun, nproc, pdir, pseudos, ecutwfc, ecutrho, degauss,
             calculation, directory, kpts=None, nbnd=40):
    from ase.calculators.espresso import Espresso, EspressoProfile
    cmd = f"{mpirun} --allow-run-as-root -np {nproc} {pw}" if nproc > 1 else pw
    profile = EspressoProfile(command=cmd, pseudo_dir=str(pdir))
    input_data = {
        "control": {"calculation": calculation, "prefix": "mat", "disk_io": "low",
                    "verbosity": "high"},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                   "smearing": "cold", "degauss": degauss, "nbnd": nbnd, "nosym": True},
        "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.2, "mixing_mode": "local-TF",
                      "electron_maxstep": 250, "startingwfc": "atomic+random"},
    }
    return Espresso(profile=profile, pseudopotentials=pseudos, input_data=input_data,
                    kpts=kpts, directory=Path(directory))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pw", required=True)
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--pseudo-dir", default="/root/phonon/pseudo")
    ap.add_argument("--ecutwfc", type=float, default=70.0)
    ap.add_argument("--ecutrho", type=float, default=280.0)
    ap.add_argument("--degauss", type=float, default=0.015)
    ap.add_argument("--nk", type=int, default=36, help="dense grid (div by 6)")
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--nbnd", type=int, default=40)
    ap.add_argument("--workdir", default="results/v100/chi_q")
    a = ap.parse_args()
    assert a.nk % 6 == 0
    m = load_mat(a.name)
    from ase.build import mx2
    kind = "2H" if m["polytype"] == "2H" else "1T"
    atoms = mx2(formula=m["formula"], kind=kind, a=m["a"], thickness=m["thickness"], vacuum=7.5)
    atoms.pbc = True
    pseudos = {m["M"]: PSEUDO[m["M"]], m["X"]: PSEUDO[m["X"]]}
    pdir = Path(a.pseudo_dir)
    work = ROOT / a.workdir / a.name
    work.mkdir(parents=True, exist_ok=True)
    nk = a.nk

    print(f"[nest:{a.name}] {m['formula']} {kind} a={m['a']}; SCF {nk//2}x{nk//2} ...", flush=True)
    t0 = time.perf_counter()
    scf = espresso(a.pw, a.mpirun, a.nproc, pdir, pseudos, a.ecutwfc, a.ecutrho,
                   a.degauss, "scf", work / "scf", kpts=(nk // 2, nk // 2, 1), nbnd=a.nbnd)
    atoms.calc = scf
    from ase.calculators.calculator import PropertyNotImplementedError
    try:
        atoms.get_potential_energy()
    except PropertyNotImplementedError:
        pass
    efermi = scf.get_fermi_level()
    print(f"[nest:{a.name}] E_F={efermi:.3f} eV ({time.perf_counter()-t0:.0f}s); nscf {nk}x{nk} ...", flush=True)

    # dense nscf on an explicit uniform grid, IN THE SCF DIR (reuse charge density);
    # explicit k-points need a weights column so ASE writes "K_POINTS crystal".
    grid = np.array([[i / nk, j / nk, 0.0] for i in range(nk) for j in range(nk)])
    grid4 = np.column_stack([grid, np.ones(len(grid))])
    nscf = espresso(a.pw, a.mpirun, a.nproc, pdir, pseudos, a.ecutwfc, a.ecutrho,
                    a.degauss, "nscf", work / "scf", kpts=grid4, nbnd=a.nbnd)
    atoms.calc = nscf
    try:
        atoms.get_potential_energy()
    except PropertyNotImplementedError:
        pass
    bands = nscf
    E = np.array([bands.get_eigenvalues(spin=0, kpt=i) for i in range(len(grid))]) - efermi
    E = E.reshape(nk, nk, -1)

    rho = np.exp(-(E / a.sigma) ** 2).sum(axis=2)
    dos_ef = float(rho.sum() / (nk * nk) / (a.sigma * np.sqrt(np.pi)))  # states/eV/cell/spin
    print(f"[nest:{a.name}] DOS(Ef) = {dos_ef:.4f} states/eV/cell/spin", flush=True)
    F = np.fft.fft2(rho)
    xi = np.fft.ifft2(np.abs(F) ** 2).real / (nk * nk)
    xi = xi / xi[0, 0]
    xi_noq0 = xi.copy(); xi_noq0[0, 0] = 0
    pk = np.unravel_index(np.argmax(xi_noq0), xi.shape)
    pk_frac = (pk[0] / nk, pk[1] / nk)
    print(f"[nest:{a.name}] GLOBAL peak (q!=0) at q=({pk_frac[0]:.3f},{pk_frac[1]:.3f}) b, "
          f"xi={xi_noq0.max():.3f}", flush=True)
    np.savez(work / f"{a.name}_nesting.npz", xi=xi, peak=np.array(pk_frac),
             xi_peak=float(xi_noq0.max()), sigma=a.sigma, nk=nk, efermi=efermi,
             name=a.name, formula=m["formula"], kind=kind)
    print(f"[nest:{a.name}] saved -> {(work/(a.name+'_nesting.npz')).relative_to(ROOT)} "
          f"({time.perf_counter()-t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
