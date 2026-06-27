"""Path-P (anharmonic distillation) Stage A for NbSe2: thermal + CDW-double-well
DFT-force data.

SSCHA (#1) showed that the *harmonic* FC distillation reproduces NbSe2's soft fc2
but the MLIP does not sustain the CDW under SSCHA fluctuations -- because F=-Phi2*u
carries no anharmonicity, so the MLIP lacks the CDW double-well. Path-P fixes this by
labelling the configurations along the CDW reaction coordinate (and around it,
thermally) with *real DFT forces*.

Sampling is MD-free on purpose: the harmonic-FT model is dynamically unstable along
the soft mode, so Langevin MD would run away. Instead we load the DFT fc2 (which gives
both the exact 3x3 supercell AND the soft eigenvector e_soft = the CDW distortion) and
generate:
  (a) double-well scan:   x0 + A * e_soft,  A in linspace(-Amax, Amax, n_scan)
  (b) thermal rattle:     x0 + Gaussian(sigma(T)),  T in --temps
  (c) rattle around the +-well minima (x0 +- A_well*e_soft + Gaussian)
Each config is DFT single-pointed (GPU pw.x, ONCV Nb/Se, cold smearing) -- identical
DFT setup to V-Q3. The rigor is in the DFT labels; the sampler only needs to cover the
relevant region. Self-contained: phonopy + ase + numpy only.

    # on the V100 (env phonon), pseudos in /root/phonon/pseudo:
    LD_LIBRARY_PATH=/root/miniconda3/envs/phonon/lib \
    /root/miniconda3/envs/phonon/bin/python scripts/path_p_nbse2_make_data.py \
        --yaml results/vq3/nbse2_dft_phonopy.yaml \
        --pw /root/gpupw.sh --pseudo-dir /root/phonon/pseudo \
        --n-scan 13 --amax 0.18 --temps 100,300 --n-therm 12 --n-well 8 \
        --workdir results/path_p_nbse2 --outdir data/path_p_nbse2
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

VASP_TO_THZ = 15.633302         # sqrt(eV/A^2/amu) -> THz
THZ_TO_CM = 33.356410           # THz -> cm^-1
KB_EV = 8.617333262e-5          # eV/K

PSEUDOS = {"Nb": "Nb_ONCV_PBE-1.2.upf", "Se": "Se_ONCV_PBE-1.2.upf"}


def soft_eigen(ph):
    """Return (atoms0, e_soft_cart[N,3] normalized to RMS=1 A, w2_all, soft_freq_cm)."""
    from ase import Atoms
    sc = ph.supercell
    atoms0 = Atoms(symbols=list(sc.symbols),
                   scaled_positions=sc.scaled_positions,
                   cell=np.array(sc.cell), pbc=[True, True, True])
    masses = atoms0.get_masses()
    N = len(atoms0)
    fc = np.asarray(ph.force_constants)          # (N,N,3,3) eV/A^2
    D = fc.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    D = 0.5 * (D + D.T)
    msqrt = np.repeat(np.sqrt(masses), 3)
    Dw = D / np.outer(msqrt, msqrt)              # mass-weighted, eV/A^2/amu
    w2, V = np.linalg.eigh(Dw)
    freq_cm = np.sign(w2) * np.sqrt(np.abs(w2)) * VASP_TO_THZ * THZ_TO_CM
    # softest (most negative w2) eigenvector -> Cartesian displacement pattern
    k = int(np.argmin(w2))
    e_cart = (V[:, k] / msqrt).reshape(N, 3)
    e_cart = e_cart / np.sqrt((e_cart ** 2).sum() / N)   # RMS atomic disp = 1 A at A=1
    return atoms0, e_cart, w2, float(freq_cm[k])


def thermal_sigma(masses, T, w2_floor_cm=60.0):
    """Per-atom isotropic Gaussian stdev (A) ~ sqrt(kT/(m*w_floor^2)); crude but the
    DFT labels carry the physics. w_floor in cm^-1 sets the rattle scale."""
    w_floor_thz = w2_floor_cm / THZ_TO_CM
    # sigma^2 = kT / (m * omega^2); convert omega(THz)->rad/s implicitly via VASP_TO_THZ units
    # work in eV/A^2/amu: k_eff = m*omega^2 with omega in THz/VASP_TO_THZ -> (eV/A^2/amu)
    keff = (w_floor_thz / VASP_TO_THZ) ** 2                # eV/A^2/amu per unit mass
    sig = np.sqrt(KB_EV * T / (keff * masses))             # A, per atom
    return sig


def make_espresso(pw, pseudo_dir, ecutwfc, ecutrho, kpts, degauss, directory):
    from ase.calculators.espresso import Espresso, EspressoProfile
    profile = EspressoProfile(command=pw, pseudo_dir=str(pseudo_dir))
    input_data = {
        "control": {"calculation": "scf", "tprnfor": True, "tstress": False,
                    "disk_io": "low", "verbosity": "low"},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho,
                   "occupations": "smearing", "smearing": "cold", "degauss": degauss},
        "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.3,
                      "electron_maxstep": 250, "diago_david_ndim": 4,
                      "startingwfc": "atomic+random"},
    }
    return Espresso(profile=profile, pseudopotentials=PSEUDOS, input_data=input_data,
                    kpts=(kpts, kpts, 1), directory=Path(directory))


def build_configs(atoms0, e_cart, masses, a):
    """Return list of (tag, ase.Atoms) configs to DFT-label."""
    rng = np.random.default_rng(a.seed)
    x0 = atoms0.get_positions()
    temps = [float(t) for t in a.temps.split(",") if t.strip()]
    configs = []

    # (a) double-well scan along the soft eigenvector
    for A in np.linspace(-a.amax, a.amax, a.n_scan):
        at = atoms0.copy(); at.set_positions(x0 + A * e_cart)
        configs.append((f"scan_A{A:+.3f}", at))

    # (b) thermal rattle around the high-symmetry structure
    for T in temps:
        sig = thermal_sigma(masses, T)[:, None]
        for j in range(a.n_therm):
            disp = rng.normal(size=x0.shape) * sig
            at = atoms0.copy(); at.set_positions(x0 + disp)
            configs.append((f"therm_T{int(T)}_{j:02d}", at))

    # (c) rattle around the +-well minima (soft coordinate displaced + thermal noise)
    for T in temps:
        sig = thermal_sigma(masses, T)[:, None]
        for sgn in (+1.0, -1.0):
            for j in range(a.n_well):
                disp = sgn * a.well * e_cart + rng.normal(size=x0.shape) * sig
                at = atoms0.copy(); at.set_positions(x0 + disp)
                configs.append((f"well_T{int(T)}_{'p' if sgn>0 else 'm'}{j:02d}", at))
    return configs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="results/vq3/nbse2_dft_phonopy.yaml")
    ap.add_argument("--pw", required=True, help="GPU pw.x launcher (e.g. /root/gpupw.sh)")
    ap.add_argument("--pseudo-dir", default="pseudo")
    ap.add_argument("--ecutwfc", type=float, default=70.0)
    ap.add_argument("--ecutrho", type=float, default=280.0)
    ap.add_argument("--kpts", type=int, default=6)
    ap.add_argument("--degauss", type=float, default=0.015)
    ap.add_argument("--n-scan", type=int, default=13, help="double-well scan points")
    ap.add_argument("--amax", type=float, default=0.18, help="max scan amplitude (A, RMS)")
    ap.add_argument("--temps", default="100,300")
    ap.add_argument("--n-therm", type=int, default=12, help="thermal rattle / T")
    ap.add_argument("--n-well", type=int, default=8, help="rattle / T / well-sign")
    ap.add_argument("--well", type=float, default=0.10, help="well displacement (A, RMS)")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N configs")
    ap.add_argument("--workdir", default="results/path_p_nbse2")
    ap.add_argument("--outdir", default="data/path_p_nbse2")
    a = ap.parse_args()

    import phonopy
    from ase.io import write

    ph = phonopy.load(a.yaml)
    atoms0, e_cart, w2, soft_cm = soft_eigen(ph)
    masses = atoms0.get_masses()
    n_imag = int((w2 < -1e-6).sum())
    print(f"[pathP-nbse2] supercell {len(atoms0)} atoms; softest mode {soft_cm:.1f} cm^-1; "
          f"n_imag(w2<0)={n_imag}", flush=True)

    configs = build_configs(atoms0, e_cart, masses, a)
    if a.limit > 0:
        configs = configs[:a.limit]
    print(f"[pathP-nbse2] {len(configs)} configs "
          f"(scan={a.n_scan}, therm={a.n_therm}/T, well={a.n_well}/T/sign, "
          f"temps={a.temps}){' [SMOKE limit %d]' % a.limit if a.limit else ''}", flush=True)

    workdir = Path(a.workdir); workdir.mkdir(parents=True, exist_ok=True)
    pseudo_dir = Path(a.pseudo_dir)
    labelled = []
    t0 = time.perf_counter()
    for i, (tag, at) in enumerate(configs):
        d = workdir / f"cfg-{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        at.calc = make_espresso(a.pw, pseudo_dir, a.ecutwfc, a.ecutrho, a.kpts,
                                a.degauss, d)
        try:
            e = float(at.get_potential_energy())
            f = np.asarray(at.get_forces())
        except Exception as exc:                 # keep going; one bad SCF != dead run
            print(f"[pathP-nbse2]   {i+1}/{len(configs)} {tag}: FAILED ({exc})", flush=True)
            continue
        at.info["REF_energy"] = e
        at.info["config_type"] = tag
        at.arrays["REF_forces"] = f
        # strip calculator so extxyz writes our REF_* not calc results
        at.calc = None
        labelled.append(at)
        dt = time.perf_counter() - t0
        print(f"[pathP-nbse2]   {i+1}/{len(configs)} {tag}: "
              f"max|F|={np.abs(f).max():.3f} eV/A  E={e:.4f}  "
              f"({dt:.0f}s, {dt/(i+1):.0f}s/cfg)", flush=True)

    if not labelled:
        print("[pathP-nbse2] no configs labelled -- aborting", flush=True)
        return 1

    # split + write extxyz (REF_energy/REF_forces are the training targets)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(labelled))
    ntest = max(1, int(round(a.test_frac * len(labelled)))) if len(labelled) > 6 else 0
    test_i = set(idx[:ntest].tolist())
    train = [labelled[j] for j in range(len(labelled)) if j not in test_i]
    test = [labelled[j] for j in range(len(labelled)) if j in test_i]

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    write(outdir / "train.xyz", train, format="extxyz")
    if test:
        write(outdir / "test.xyz", test, format="extxyz")
    fall = np.concatenate([at.arrays["REF_forces"].ravel() for at in labelled])
    print(f"[pathP-nbse2] wrote {len(train)} train + {len(test)} test -> {outdir}", flush=True)
    print(f"[pathP-nbse2] <|F_dft|>_rms = {np.sqrt((fall**2).mean())*1000:.0f} meV/A "
          f"over {len(labelled)} configs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
