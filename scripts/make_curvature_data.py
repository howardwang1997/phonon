"""Curvature-aware 3rd-order distillation data (P2: turn the Task-3 negative
into a positive).

The naive anharmonic distillation regressed κ because fine-tuning on large-
displacement cubic-label configs degraded the small-displacement phonon Hessian
(Φ₂) that κ depends on. Fix: a JOINT dataset that strongly anchors the harmonic
Hessian while adding the cubic correction:

  • harmonic-anchor configs: SMALL rattle, labels from Φ₂ ONLY
      F = -Φ₂u ,  E = ½ uΦ₂u                 (pins the curvature; up-weighted)
  • anharmonic configs:      LARGER rattle, labels from the full cubic expansion
      F = -Φ₂u - ½ Φ₃:uu ,  E = ½uΦ₂u + (1/6)Φ₃uuu   (adds anharmonicity)

Harmonic configs dominate by count and by per-config weight, so the Hessian is
preserved while Φ₃ information is still injected.

    python scripts/make_curvature_data.py --material Si --fc-dir results/fc3 \
        --out data/curv_Si --n-harm 450 --n-anharm 150 --harm-weight 3.0
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
    ap.add_argument("--out", default="data/curv")
    ap.add_argument("--n-harm", type=int, default=450, help="harmonic-anchor configs (small rattle)")
    ap.add_argument("--n-anharm", type=int, default=150, help="anharmonic configs (large rattle)")
    ap.add_argument("--harm-weight", type=float, default=3.0, help="per-config weight on harmonic anchors")
    ap.add_argument("--harm-amps", type=float, nargs="+", default=[0.01, 0.02, 0.03])
    ap.add_argument("--anharm-amps", type=float, nargs="+", default=[0.05, 0.07, 0.09])
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

    def hforce(u):
        return -np.einsum("ijab,jb->ia", fc2, u)

    def henergy(u):
        return 0.5 * np.einsum("ia,ijab,jb->", u, fc2, u)

    def aforce(u):
        return hforce(u) - 0.5 * np.einsum("ijkabc,jb,kc->ia", fc3, u, u)

    def aenergy(u):
        return henergy(u) + (1.0 / 6.0) * np.einsum("ia,jb,kc,ijkabc->", u, u, u, fc3)

    rng = np.random.default_rng(args.seed)
    configs = []

    def add(u, anharm, weight):
        at = base.copy(); at.set_positions(pos0 + u)
        E = float(aenergy(u) if anharm else henergy(u))
        F = aforce(u) if anharm else hforce(u)
        at.info["REF_energy"] = E; at.info["energy"] = E
        at.arrays["REF_forces"] = F; at.arrays["forces"] = F
        at.info["config_weight"] = float(weight)        # MACE per-config weight
        at.info["mp_id"] = args.material
        at.info["kind"] = "harm" if not anharm else "anharm"
        configs.append(at)

    # equilibrium anchor (harmonic, heavy weight)
    add(np.zeros((N, 3)), anharm=False, weight=args.harm_weight)
    # harmonic anchors: small rattle, Φ₂-only labels, up-weighted
    for _ in range(args.n_harm):
        s = args.harm_amps[int(rng.integers(len(args.harm_amps)))]
        add(rng.normal(0.0, s, (N, 3)), anharm=False, weight=args.harm_weight)
    # small single-atom harmonic probes (directly supervise Φ₂ columns)
    for i in rng.choice(N, size=min(N, 8), replace=False):
        for a in range(3):
            for delta in (0.01, -0.01, 0.02, -0.02):
                u = np.zeros((N, 3)); u[i, a] = delta
                add(u, anharm=False, weight=args.harm_weight)
    # anharmonic configs: larger rattle, full cubic labels, weight 1
    for _ in range(args.n_anharm):
        s = args.anharm_amps[int(rng.integers(len(args.anharm_amps)))]
        add(rng.normal(0.0, s, (N, 3)), anharm=True, weight=1.0)
    for i in rng.choice(N, size=min(N, 6), replace=False):
        for a in range(3):
            for delta in (0.06, -0.06, 0.09, -0.09):
                u = np.zeros((N, 3)); u[i, a] = delta
                add(u, anharm=True, weight=1.0)

    rng.shuffle(configs)
    nval = int(len(configs) * args.val_frac)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write(out / "val.xyz", configs[:nval], format="extxyz")
    write(out / "train.xyz", configs[nval:], format="extxyz")
    nh = sum(1 for c in configs if c.info["kind"] == "harm")
    print(f"wrote {len(configs)-nval} train + {nval} val  ({nh} harmonic-anchor + "
          f"{len(configs)-nh} anharmonic, harm weight {args.harm_weight}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
