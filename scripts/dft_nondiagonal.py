"""Line B factor: non-diagonal supercells (Lloyd-Williams & Monserrat, PRB 2015).

To get the dynamical matrix at a commensurate q = (h/n,k/n,l/n), the supercell
must be commensurate with q (S^T q in Z^3). A *diagonal* N×N×N supercell reaches
the whole N³ q-grid but costs N³ primitive cells. A *non-diagonal* supercell can
be commensurate with a single q at far smaller |det(S)|. Computing each
inequivalent q with its own minimal non-diagonal supercell costs
sum_q |det(S_q)| primitive cells << N³.

This measures that atom-count (=SCF-cost) reduction factor for standard q-grids.
Pure integer search (no DFT). Validates against the q-grids a finite-displacement
phonon would otherwise need a big diagonal supercell for.
"""
from __future__ import annotations

import argparse
import itertools
from fractions import Fraction
from math import gcd

import numpy as np


def min_nondiagonal_det(q):
    """Smallest |det(S)| over integer matrices S with S^T q in Z^3 (q reduced coords).
    Uses Hermite normal form: upper-triangular S with positive diagonal; the lcm of
    the q-component denominators bounds the search."""
    fr = [Fraction(x).limit_denominator(10000) for x in q]
    dens = [f.denominator for f in fr]
    n = 1
    for d in dens:
        n = n * d // gcd(n, d)               # lcm of denominators
    # search HNF matrices of increasing determinant up to n
    best = None
    for det in range(1, n + 1):
        if n % det:
            continue
        # enumerate upper-triangular integer matrices with this determinant
        for a in divisors(det):
            for d in divisors(det // a):
                f = det // (a * d)
                for b in range(a):
                    for c in range(a):
                        for e in range(d):
                            S = np.array([[a, b, c], [0, d, e], [0, 0, f]])
                            v = S.T @ np.array([float(x) for x in fr])
                            if np.allclose(v, np.round(v), atol=1e-6):
                                return int(round(abs(np.linalg.det(S))))
    return n


def divisors(m):
    return [i for i in range(1, m + 1) if m % i == 0]


def qgrid(N):
    """inequivalent commensurate q-points of an N×N×N Monkhorst-Pack-like grid (reduced)."""
    seen = set()
    out = []
    for h, k, l in itertools.product(range(N), repeat=3):
        q = (Fraction(h, N), Fraction(k, N), Fraction(l, N))
        # fold by inversion symmetry q ~ -q
        nq = tuple(((-x) % 1) for x in q)
        key = min((tuple(q), nq))
        if key in seen:
            continue
        seen.add(key)
        out.append([float(x) for x in q])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--natoms-prim", type=int, default=2, help="primitive-cell atoms (Si=2)")
    args = ap.parse_args()
    # DFT (plane-wave diagonalization) scales ~ cubically with atom count, so the
    # cost is dominated by the LARGEST single cell, not the total atom count.
    print("q-grid | peak cell (diag -> non-diag) | peak-size factor | SCF-cost factor (~atoms^3)")
    for N in args.grids:
        qs = qgrid(N)
        dets = [min_nondiagonal_det(q) for q in qs]
        peak_diag = N ** 3                      # one N^3 diagonal supercell (the only cell)
        peak_nd = max(dets)                     # largest single non-diagonal cell
        cost_diag = peak_diag ** 3              # one big SCF, ~atoms^3
        cost_nd = sum(d ** 3 for d in dets)     # many small SCFs, each ~atoms^3
        print(f"  {N}^3   |   {peak_diag:5d} -> {peak_nd:3d} cells       |   {peak_diag/peak_nd:6.1f}x      "
              f"|  {cost_diag/cost_nd:8.1f}x")
    print("\nPeak supercell shrinks ~N^3 -> ~N; since DFT cost ~atoms^3 and is dominated by the "
          "largest cell,\nthe non-diagonal set is dramatically cheaper for exact fine-q phonons "
          "(grows steeply with q-density).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
