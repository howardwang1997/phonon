from __future__ import annotations

import numpy as np

from . import symmetry as sym


def _cart_rotation(structure, R_frac):
    A = structure.cell
    return A @ R_frac @ np.linalg.inv(A)


def _forward_atom_map(structure, R, tau, symprec=1e-5):
    N = structure.natoms
    frac = structure.positions
    fmap = np.full(N, -1, dtype=int)
    for l in range(N):
        img = R @ frac[l] + tau
        for m in range(N):
            if structure.species[m] != structure.species[l]:
                continue
            diff = img - frac[m]
            diff = diff - np.round(diff)
            if np.linalg.norm(diff) < symprec:
                fmap[l] = m
                break
    return fmap


def assemble_from_forces(structure, plan, forces, symprec=1e-5):
    from .displacements import flatten_evals
    evals = flatten_evals(plan)
    if forces.shape[0] != len(evals):
        raise ValueError(
            f"forces first dim {forces.shape[0]} != num evals {len(evals)}"
        )
    N = structure.natoms
    FC = np.zeros((3 * N, 3 * N))
    eidx = 0
    for task in plan:
        i = task.center
        dirs = task.directions
        C = np.zeros((N, 3, dirs.shape[0]))
        for a, direction in enumerate(dirs):
            Fp = forces[eidx].reshape(N, 3)
            eidx += 1
            Fm = forces[eidx].reshape(N, 3)
            eidx += 1
            C[:, :, a] = -(Fp - Fm) / (2.0 * task.distance)
        D = dirs
        Dinv = np.linalg.pinv(D)
        Phi_i = C @ Dinv.T
        FC[:, 3 * i:3 * i + 3] = Phi_i.reshape(3 * N, 3)
        for (j, R, tau) in task.orbit:
            if j == i:
                continue
            Rcart = _cart_rotation(structure, R)
            fmap = _forward_atom_map(structure, R, tau, symprec)
            if (fmap < 0).any():
                raise RuntimeError("incomplete symmetry map; raise symprec")
            for l in range(N):
                m = fmap[l]
                block = Phi_i[l]
                rotated = Rcart @ block @ Rcart.T
                FC[3 * m:3 * m + 3, 3 * j:3 * j + 3] = rotated
    FC = symmetrize(FC)
    FC = enforce_acoustic_sum_rule(FC)
    return FC


def assemble_force_constants(structure, plan, force_callable, symprec=1e-5):
    N = structure.natoms
    FC = np.zeros((3 * N, 3 * N))
    for task in plan:
        i = task.center
        dirs = task.directions
        C = np.zeros((N, 3, dirs.shape[0]))
        for a, direction in enumerate(dirs):
            up = +task.distance * direction
            dn = -task.distance * direction
            Fp = np.asarray(force_callable(i, up)).reshape(N, 3)
            Fm = np.asarray(force_callable(i, dn)).reshape(N, 3)
            C[:, :, a] = -(Fp - Fm) / (2.0 * task.distance)
        D = dirs
        Dinv = np.linalg.pinv(D)
        Phi_i = C @ Dinv.T
        FC_i = Phi_i.reshape(3 * N, 3)
        FC[:, 3 * i:3 * i + 3] = FC_i
        for (j, R, tau) in task.orbit:
            if j == i:
                continue
            Rcart = _cart_rotation(structure, R)
            fmap = _forward_atom_map(structure, R, tau, symprec)
            if (fmap < 0).any():
                raise RuntimeError("incomplete symmetry map; raise symprec")
            for l in range(N):
                m = fmap[l]
                block = Phi_i[l]
                rotated = Rcart @ block @ Rcart.T
                FC[3 * m:3 * m + 3, 3 * j:3 * j + 3] = rotated
    FC = symmetrize(FC)
    FC = enforce_acoustic_sum_rule(FC)
    return FC


def symmetrize(fc):
    return 0.5 * (fc + fc.T)


def enforce_acoustic_sum_rule(fc):
    n3 = fc.shape[0]
    N = n3 // 3
    fc = fc.copy()
    for _ in range(20):
        g = fc.reshape(N, 3, N, 3)
        drift = g.sum(axis=2)
        g = g - drift[:, :, None, :] / N
        drift2 = g.sum(axis=0)
        g = g - drift2[None, :, :, :] / N
        fc = symmetrize(g.reshape(n3, n3))
        max_drift = abs(fc.reshape(N, 3, N, 3).sum(axis=2)).max()
        if max_drift < 1e-9:
            break
    return fc


def mass_weights(structure):
    N = structure.natoms
    m = structure.masses
    inv_sqrt = np.repeat(1.0 / np.sqrt(m), 3)
    M = np.diag(inv_sqrt)
    return M


def mass_weight_fc(fc, structure):
    M = mass_weights(structure)
    return M @ fc @ M


def un_mass_weight_fc(dyn, structure):
    m = structure.masses
    sqrt = np.repeat(np.sqrt(m), 3)
    return (np.diag(sqrt) @ dyn @ np.diag(sqrt))
