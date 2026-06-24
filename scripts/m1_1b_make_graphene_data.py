"""M1.1b step 2: build graphene FC-distillation data from the DFT fc2.

Loads the phonopy object saved by ``m1_1b_graphene_dft.py`` (graphene's own
DFPT-quality fc2), then synthesises harmonic-labelled rattled supercells
(F = -Phi u, E = 1/2 u Phi u) exactly as the bulk FC-distillation does -- this is
the *graphene-specific* training set whose whole point is to inject the correct
Gamma/K curvature the bulk-trained model misses.

    conda run -n phonon python scripts/m1_1b_make_graphene_data.py \
        --phonopy results/m1_1b/dft/graphene_dft_phonopy.yaml \
        --out data/finetune_graphene --n-configs 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import phonopy
from ase.io import write

from phonon_accel.finetune.dataset import harmonic_energy, harmonic_forces
from phonon_accel.phonons import phonopy_to_ase


def configs_from_phonopy(ph, n_configs, rattle_std, n_single_sites,
                         single_deltas=(0.01, -0.01, 0.03), seed=0, label="graphene"):
    fc = np.asarray(ph.force_constants)
    base = phonopy_to_ase(ph.supercell)
    N = len(base)
    if fc.shape[:2] != (N, N):
        raise ValueError(f"need full fc (N,N,3,3) with N={N}, got {fc.shape}")
    pos0 = base.get_positions()
    rng = np.random.default_rng(seed)
    configs = []

    def _add(u):
        at = base.copy()
        at.set_positions(pos0 + u)
        F = harmonic_forces(fc, u)
        E = harmonic_energy(fc, u)
        at.info["energy"] = E
        at.info["REF_energy"] = E
        at.arrays["forces"] = F
        at.arrays["REF_forces"] = F
        at.info["material"] = label
        configs.append(at)

    _add(np.zeros((N, 3)))                                   # equilibrium anchor
    for _ in range(n_configs):                               # random rattles
        _add(rng.normal(0.0, rattle_std, (N, 3)))
    for i in rng.choice(N, size=min(N, n_single_sites), replace=False):
        for a in range(3):                                   # single-atom probes
            for delta in single_deltas:
                u = np.zeros((N, 3))
                u[i, a] = delta
                _add(u)
    return configs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phonopy", required=True, help="phonopy yaml with force constants")
    ap.add_argument("--out", default="data/finetune_graphene")
    ap.add_argument("--n-configs", type=int, default=60)
    ap.add_argument("--rattle-std", type=float, default=0.04)
    ap.add_argument("--n-single-sites", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ph = phonopy.load(str(ROOT / a.phonopy) if not Path(a.phonopy).is_absolute()
                      else a.phonopy, is_compact_fc=False)
    fc = np.asarray(ph.force_constants)
    print(f"loaded {a.phonopy}: fc shape {fc.shape}, supercell {len(phonopy_to_ase(ph.supercell))} atoms")

    cfgs = configs_from_phonopy(ph, a.n_configs, a.rattle_std, a.n_single_sites,
                                seed=a.seed)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(cfgs)
    n_val = int(len(cfgs) * a.val_fraction)
    val, train = cfgs[:n_val], cfgs[n_val:]

    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "train.xyz", train, format="extxyz")
    write(out / "val.xyz", val, format="extxyz")
    fmax = max(float(np.abs(c.arrays["forces"]).max()) for c in cfgs)
    summary = {"n_train": len(train), "n_val": len(val), "n_total": len(cfgs),
               "rattle_std": a.rattle_std, "max_force_eVA": round(fmax, 3),
               "source": a.phonopy}
    (out / "split.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out/'train.xyz'} + {out/'val.xyz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
