import numpy as np

from phonongpu.structure import Structure
from phonongpu.displacements import displacement_plan, count_force_evaluations, flatten_evals
from phonongpu.force_constants import (
    assemble_force_constants,
    assemble_from_forces,
    enforce_acoustic_sum_rule,
)


def _low_sym():
    cell = np.array([[4.1, 0, 0], [0.2, 4.3, 0], [0.1, 0.2, 4.2]])
    return Structure(cell, [[0, 0, 0], [0.31, 0.34, 0.37]], ["A", "B"], [10.0, 20.0])


def _spring_fc(struct, k, cut):
    pairs = struct.neighbors(cut)
    H = np.zeros((3 * struct.natoms, 3 * struct.natoms))
    for (i, j, sh, dist, vec) in pairs:
        if dist < 1e-8:
            continue
        n = vec / dist
        oo = np.outer(n, n) * k
        H[3 * i:3 * i + 3, 3 * i:3 * i + 3] += oo
        H[3 * j:3 * j + 3, 3 * j:3 * j + 3] += oo
        H[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= oo
        H[3 * j:3 * j + 3, 3 * i:3 * i + 3] -= oo
    return enforce_acoustic_sum_rule(H)


def test_fc_recovery_low_symmetry():
    rng = np.random.default_rng(42)
    s = _low_sym()
    N = s.natoms
    A = rng.standard_normal((3 * N, 3 * N))
    FC = enforce_acoustic_sum_rule(A + A.T)
    plan = displacement_plan(s, distance=0.02)

    def force(idx, u):
        return -FC[:, 3 * idx:3 * idx + 3] @ u

    rec = assemble_force_constants(s, plan, force)
    assert np.abs(FC - rec).max() < 1e-10
    assert np.allclose(rec, rec.T, atol=1e-9)


def test_fc_recovery_symmetry_reduced():
    a = 5.43
    cell = a * np.eye(3)
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    s = Structure(cell, np.vstack([fcc, fcc + 0.25]), ["Si"] * 8)
    H = _spring_fc(s, 1.0, np.sqrt(3) / 4 * a * 1.05)
    plan = displacement_plan(s, distance=0.02)
    assert count_force_evaluations(plan) < 6 * s.natoms
    evals = flatten_evals(plan)

    def force(idx, u):
        return -H[:, 3 * idx:3 * idx + 3] @ u

    forces = np.array([force(idx, u) for (idx, u) in evals])
    rec = assemble_from_forces(s, plan, forces)
    assert np.abs(H - rec).max() < 1e-8
    drift = abs(rec.reshape(s.natoms, 3, s.natoms, 3).sum(axis=2)).max()
    assert drift < 1e-9
