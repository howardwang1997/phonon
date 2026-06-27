"""B-rigorous: NbSe2 Fermi-surface nesting function xi(q) from a dense DFT grid.

xi(q) = (1/N) sum_k rho(k) rho(k+q),  rho(k) = sum_n Gaussian(eps_nk - E_F, sigma)
      = autocorrelation of the Fermi-surface density map -> via FFT (fast).

A nesting-driven CDW shows xi(q) PEAKING at q_CDW. For monolayer NbSe2,
q_CDW = (1/3, 0) (3x3) = 1/3 of Gamma-...-Gamma along b1 (i.e. 2/3 of Gamma-M).
If xi(q) does NOT peak there, the CDW is not nesting-driven (momentum-dependent
e-ph coupling instead) -- the rigorous version of the crude k_F check in
vq3c_nbse2_bands.py.

    python scripts/vq3d_nbse2_nesting.py --pw /root/gpupw.sh --nproc 1 \
        --pseudo-dir pseudo --a 3.474 --nk 36 --sigma 0.12
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import td_common as tdc

PSEUDOS = {"Nb": "Nb_ONCV_PBE-1.2.upf", "Se": "Se_ONCV_PBE-1.2.upf"}


def espresso(pw, mpirun, nproc, pdir, ecutwfc, ecutrho, degauss, calculation,
             directory, kpts=None, nbnd=28):
    from ase.calculators.espresso import Espresso, EspressoProfile
    cmd = f"{mpirun} --allow-run-as-root -np {nproc} {pw}" if nproc > 1 else pw
    profile = EspressoProfile(command=cmd, pseudo_dir=str(pdir))
    input_data = {
        "control": {"calculation": calculation, "prefix": "nbse2", "disk_io": "low",
                    "verbosity": "high"},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                   "smearing": "cold", "degauss": degauss, "nbnd": nbnd, "nosym": True},
        "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.3, "electron_maxstep": 250,
                      "diago_david_ndim": 4, "startingwfc": "atomic+random"},
    }
    return Espresso(profile=profile, pseudopotentials=PSEUDOS, input_data=input_data,
                    kpts=kpts, directory=Path(directory))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw", required=True)
    ap.add_argument("--mpirun", default="")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--pseudo-dir", default="pseudo")
    ap.add_argument("--a", type=float, default=3.474)
    ap.add_argument("--thickness", type=float, default=3.361)
    ap.add_argument("--ecutwfc", type=float, default=70.0)
    ap.add_argument("--ecutrho", type=float, default=280.0)
    ap.add_argument("--degauss", type=float, default=0.015)
    ap.add_argument("--nk", type=int, default=36, help="dense grid (div by 6 so M,K,q_CDW are grid pts)")
    ap.add_argument("--sigma", type=float, default=0.12, help="FS Gaussian width (eV)")
    ap.add_argument("--nbnd", type=int, default=28)
    ap.add_argument("--workdir", default="results/vq3d")
    a = ap.parse_args()
    mp = a.mpirun or "mpirun"
    pdir = (ROOT / a.pseudo_dir) if not Path(a.pseudo_dir).is_absolute() else Path(a.pseudo_dir)
    work = ROOT / a.workdir
    work.mkdir(parents=True, exist_ok=True)
    atoms = tdc.build_monolayer("nbse2", a=a.a, thickness=a.thickness)
    atoms.wrap()
    nk = a.nk
    assert nk % 6 == 0, "nk must be divisible by 6"

    # 1) SCF for the density + E_F
    print(f"[nest] NbSe2 a={a.a}; SCF {nk//2}x{nk//2} for density ...", flush=True)
    t0 = time.perf_counter()
    scf = espresso(a.pw, mp, a.nproc, pdir, a.ecutwfc, a.ecutrho, a.degauss, "scf",
                   work / "scf", kpts=(nk // 2, nk // 2, 1), nbnd=a.nbnd)
    atoms.calc = scf
    atoms.get_potential_energy()
    efermi = scf.get_fermi_level()
    print(f"[nest] SCF done ({time.perf_counter()-t0:.0f}s), E_F={efermi:.3f} eV", flush=True)

    # 2) eigenvalues on the FULL nk x nk grid (explicit, nosym) for the autocorrelation
    grid = np.array([[i / nk, j / nk, 0.0] for i in range(nk) for j in range(nk)])
    grid4 = np.column_stack([grid, np.ones(len(grid))])
    print(f"[nest] eigenvalues on full {nk}x{nk} grid ({len(grid)} k) ...", flush=True)
    bands = espresso(a.pw, mp, a.nproc, pdir, a.ecutwfc, a.ecutrho, a.degauss, "bands",
                     work / "scf", kpts=grid4, nbnd=a.nbnd)
    atoms.calc = bands
    from ase.calculators.calculator import PropertyNotImplementedError
    try:
        atoms.get_potential_energy()
    except PropertyNotImplementedError:
        pass
    E = np.array([bands.get_eigenvalues(spin=0, kpt=i) for i in range(len(grid))]) - efermi
    E = E.reshape(nk, nk, -1)

    # 3) FS density map rho(k) = sum_n Gaussian(eps, sigma); xi(q) = autocorrelation (FFT)
    rho = np.exp(-(E / a.sigma) ** 2).sum(axis=2)  # (nk, nk)
    F = np.fft.fft2(rho)
    xi = np.fft.ifft2(np.abs(F) ** 2).real / (nk * nk)  # xi[qi,qj], xi[0,0] = max self
    xi = xi / xi[0, 0]  # normalize to q=0

    # 4) extract along Gamma-M-K-Gamma (grid indices); q_CDW = (nk/3, 0)
    def gidx(fi, fj):
        return int(round(fi * nk)) % nk, int(round(fj * nk)) % nk
    path = []
    labels = []
    # G(0,0) -> M(1/2,0) -> K(1/3,1/3) -> G
    segs = [((0, 0), (0.5, 0.0), "G", "M"), ((0.5, 0.0), (1 / 3, 1 / 3), "M", "K"),
            ((1 / 3, 1 / 3), (0, 0), "K", "G")]
    pathvals, pathx, lab_pos = [], [], [0]
    x = 0.0
    for (p0, p1, l0, l1) in segs:
        for t in np.linspace(0, 1, 60, endpoint=False):
            fi = p0[0] + t * (p1[0] - p0[0]); fj = p0[1] + t * (p1[1] - p0[1])
            qi, qj = gidx(fi, fj)
            pathvals.append(xi[qi, qj]); pathx.append(x); x += 1
        lab_pos.append(x)
    pathvals = np.array(pathvals)

    # q_CDW = (1/3, 0) on Gamma-M; report xi there and the global peak (excluding q=0)
    qi_cdw, qj_cdw = gidx(1 / 3, 0)
    xi_cdw = xi[qi_cdw, qj_cdw]
    xi_noq0 = xi.copy(); xi_noq0[0, 0] = 0
    pk = np.unravel_index(np.argmax(xi_noq0), xi.shape)
    pk_frac = (pk[0] / nk, pk[1] / nk)
    print(f"[nest] sigma={a.sigma} eV, grid {nk}x{nk}", flush=True)
    print(f"[nest] xi(q_CDW=(1/3,0)) = {xi_cdw:.3f}  (normalized to xi(0)=1)", flush=True)
    print(f"[nest] GLOBAL peak (q!=0) at q=({pk_frac[0]:.3f},{pk_frac[1]:.3f}) b, xi={xi_noq0.max():.3f}",
          flush=True)
    print(f"[nest] => CDW nesting-driven? {'PLAUSIBLE' if xi_cdw > 0.7*xi_noq0.max() and (abs(pk_frac[0]-1/3)<0.08 and pk_frac[1]<0.08) else 'NO (xi does not peak at q_CDW)'}",
          flush=True)

    out = work / "nbse2_nesting.npz"
    np.savez(out, xi=xi, pathvals=pathvals, pathx=np.array(pathx),
             label_positions=np.array(lab_pos), labels=np.array(["G", "M", "K", "G"]),
             xi_cdw=xi_cdw, peak=np.array(pk_frac), sigma=a.sigma, nk=nk, efermi=efermi)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(pathx, pathvals, color="tab:blue", lw=1.6)
    qcdw_x = lab_pos[1] * (1 / 3) / 0.5  # 1/3 along G-M (M at 1/2)
    ax.axvline(qcdw_x, color="tab:red", ls="--", lw=1.2, label="q$_{CDW}$ = 1/3 b$_1$")
    for xp in lab_pos:
        ax.axvline(xp, color="0.8", lw=0.5)
    ax.set_xticks(lab_pos); ax.set_xticklabels(["Γ", "M", "K", "Γ"])
    ax.set_xlim(pathx[0], lab_pos[-1])
    ax.set_ylabel("nesting function ξ(q) / ξ(0)")
    ax.set_title("NbSe$_2$ Fermi-surface nesting ξ(q): does it peak at q$_{CDW}$?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figp = ROOT / "results/figures/nbse2_nesting.png"
    fig.savefig(figp, dpi=160)
    print(f"[nest] saved -> {out.relative_to(ROOT)}, {figp.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
