"""3rd-order (anharmonic) FC distillation data. Extends the harmonic FC
distillation: labels rattled supercells with the FULL cubic Taylor expansion
  E(u) = ½ uΦ₂u + (1/6) Φ₃uuu
  F_i  = -Φ₂u - ½ Φ₃:uu
using DFT-derived Φ₂ (fc2) and Φ₃ (fc3) from dft_fc3.py. Fine-tuning an MLIP on
these teaches the anharmonic response (the part harmonic distillation misses) ->
target the κ residual.

    python scripts/make_anharm_data.py --material Si --fc-dir results/fc3 \
        --out data/anharm_Si --n-configs 200 --rattle-std 0.06
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms
from ase.io import write


def load_fc(path, keys):
    with h5py.File(path, "r") as f:
        for k in keys:
            if k in f:
                return f[k][:]
    raise KeyError(f"none of {keys} in {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="Si")
    ap.add_argument("--fc-dir", default="results/fc3")
    ap.add_argument("--out", default="data/anharm")
    ap.add_argument("--n-configs", type=int, default=200)
    ap.add_argument("--rattle-std", type=float, default=0.06,
                    help="larger than harmonic (0.02) to excite anharmonicity")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = Path(args.fc_dir)
    fc2 = load_fc(d / f"{args.material}_fc2.hdf5", ["fc2", "force_constants"])      # (N,N,3,3)
    fc3 = load_fc(d / f"{args.material}_fc3.hdf5", ["fc3"])                          # (N,N,N,3,3,3)
    cells = np.load(d / f"{args.material}_cells.npz")
    base = Atoms(numbers=cells["sc_numbers"], scaled_positions=cells["sc_positions"],
                 cell=cells["sc_cell"], pbc=True)
    N = len(base); pos0 = base.get_positions()
    print(f"[{args.material}] supercell {N} atoms, fc2 {fc2.shape}, fc3 {fc3.shape}")

    def aforce(u):
        return -np.einsum("ijab,jb->ia", fc2, u) - 0.5 * np.einsum("ijkabc,jb,kc->ia", fc3, u, u)

    def aenergy(u):
        return (0.5 * np.einsum("ia,ijab,jb->", u, fc2, u)
                + (1.0 / 6.0) * np.einsum("ia,jb,kc,ijkabc->", u, u, u, fc3))

    rng = np.random.default_rng(args.seed)
    configs = []

    def add(u):
        at = base.copy(); at.set_positions(pos0 + u)
        E = float(aenergy(u)); F = aforce(u)
        at.info["REF_energy"] = E; at.info["energy"] = E
        at.arrays["REF_forces"] = F; at.arrays["forces"] = F
        at.info["mp_id"] = args.material
        configs.append(at)

    add(np.zeros((N, 3)))                                   # equilibrium anchor
    for _ in range(args.n_configs):                        # random rattles (anharmonic regime)
        add(rng.normal(0.0, args.rattle_std, (N, 3)))
    # single-atom probes at a few amplitudes (directly excite Φ₃ columns)
    for i in rng.choice(N, size=min(N, 6), replace=False):
        for a in range(3):
            for delta in (0.03, -0.03, 0.08, -0.08):
                u = np.zeros((N, 3)); u[i, a] = delta; add(u)

    rng.shuffle(configs)
    nval = int(len(configs) * args.val_frac)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write(out / "val.xyz", configs[:nval], format="extxyz")
    write(out / "train.xyz", configs[nval:], format="extxyz")
    print(f"wrote {len(configs)-nval} train + {nval} val anharmonic configs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
