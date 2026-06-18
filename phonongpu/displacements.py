from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from . import symmetry as sym


@dataclass
class DisplacementAtom:
    center: int
    distance: float
    orbit: list
    directions: np.ndarray = field(default_factory=lambda: np.eye(3))

    @property
    def num_evals(self):
        return 2 * self.directions.shape[0]

    def eval_vectors(self):
        out = []
        for d in self.directions:
            out.append((self.center, +self.distance * d, +1))
            out.append((self.center, -self.distance * d, -1))
        return out


def site_symmetry_reduced_directions(structure, atom_index, symprec=1e-5):
    site = sym.site_symmetry(structure, atom_index, symprec)
    axes = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    reduced = []
    for a in axes:
        if any(_parallel(a, r) for r in reduced):
            continue
        orbit_seen = False
        for R, _ in site:
            Ra = R @ a
            for r in reduced:
                if _parallel(Ra, r):
                    orbit_seen = True
                    break
            if orbit_seen:
                break
        if not orbit_seen:
            reduced.append(a)
    return np.array(reduced) if reduced else np.eye(3)


def _parallel(a, b, tol=1e-4):
    a = a / (np.linalg.norm(a) + 1e-30)
    b = b / (np.linalg.norm(b) + 1e-30)
    return min(np.linalg.norm(a - b), np.linalg.norm(a + b)) < tol


def displacement_plan(structure, distance=0.01, symprec=1e-5, reduce_directions=False):
    orbits = sym.atom_orbits(structure, symprec)
    plan = []
    for orbit in orbits:
        center = orbit[0][0]
        if reduce_directions:
            dirs = site_symmetry_reduced_directions(structure, center, symprec)
            if dirs.shape[0] != 3:
                dirs = np.eye(3)
        else:
            dirs = np.eye(3)
        plan.append(DisplacementAtom(center, distance, orbit, dirs))
    return plan


def count_force_evaluations(plan):
    return sum(d.num_evals for d in plan)


def flatten_evals(plan):
    evals = []
    for task in plan:
        center = task.center
        for direction in task.directions:
            evals.append((center, +task.distance * direction))
            evals.append((center, -task.distance * direction))
    return evals
