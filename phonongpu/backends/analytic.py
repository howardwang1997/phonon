from __future__ import annotations

import numpy as np

from .base import ForceBackend
from ..structure import Structure
from ..force_constants import enforce_acoustic_sum_rule


class AnalyticForceBackend(ForceBackend):
    name = "analytic"

    def __init__(self, fc=None, spring_k=1.0, spring_cutoff_factor=1.05, options=None):
        super().__init__(options)
        self._fc = fc
        self.spring_k = spring_k
        self.spring_cutoff_factor = spring_cutoff_factor

    @classmethod
    def from_springs(cls, structure, k=1.0, cutoff_factor=1.05):
        pairs = structure.neighbors(structure.cell.diagonal() * 0 + 1.0)
        cutoff = _auto_nn_cutoff(structure, cutoff_factor)
        H = _spring_fc(structure, k, cutoff)
        obj = cls(fc=H)
        obj._cutoff = cutoff
        return obj

    def _ensure_fc(self, structure):
        if self._fc is None or self._fc.shape[0] != 3 * structure.natoms:
            cutoff = _auto_nn_cutoff(structure, self.spring_cutoff_factor)
            self._fc = _spring_fc(structure, self.spring_k, cutoff)
        return self._fc

    def evaluate(self, structure, evals, executor=None):
        H = self._ensure_fc(structure)
        N = structure.natoms
        forces = np.zeros((len(evals), N, 3))
        for k, (idx, vec) in enumerate(evals):
            col = H[:, 3 * idx:3 * idx + 3]
            forces[k] = (-col @ np.asarray(vec)).reshape(N, 3)
        return forces


def _auto_nn_cutoff(structure, factor=1.05):
    cart = structure.cartesian()
    n = structure.natoms
    a = structure.cell
    lens = np.linalg.norm(a, axis=1)
    r_ext = [int(np.ceil(max(lens) / lens[i])) + 1 for i in range(3)]
    dmin = 1e18
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for ix in range(-r_ext[0], r_ext[0] + 1):
                for iy in range(-r_ext[1], r_ext[1] + 1):
                    for iz in range(-r_ext[2], r_ext[2] + 1):
                        shift = a @ np.array([ix, iy, iz], dtype=float)
                        d = np.linalg.norm(cart[j] + shift - cart[i])
                        if 1e-6 < d < dmin:
                            dmin = d
    return factor * dmin


def _spring_fc(structure, k, cutoff):
    pairs = structure.neighbors(cutoff)
    n3 = 3 * structure.natoms
    H = np.zeros((n3, n3))
    for (i, j, sh, dist, vec) in pairs:
        if dist < 1e-8:
            continue
        nrm = vec / dist
        oo = np.outer(nrm, nrm) * k
        H[3 * i:3 * i + 3, 3 * i:3 * i + 3] += oo
        H[3 * j:3 * j + 3, 3 * j:3 * j + 3] += oo
        H[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= oo
        H[3 * j:3 * j + 3, 3 * i:3 * i + 3] -= oo
    return enforce_acoustic_sum_rule(H)
