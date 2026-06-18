from __future__ import annotations

import numpy as np

from .structure import Structure
from .displacements import displacement_plan, flatten_evals, count_force_evaluations
from .force_constants import assemble_from_forces
from .dynamical import DynamicalMatrix
from .phonon import PhononBands, PhononDOS
from .backends import get_backend
from .scheduler import MultiGPUExecutor


class PhononResult:
    def __init__(self, primitive, supercell, fc, dyn):
        self.primitive = primitive
        self.supercell = supercell
        self.force_constants = fc
        self.dynamical = dyn
        self.bands = PhononBands(dyn)
        self.dos = PhononDOS(dyn)

    @property
    def gamma_frequencies(self):
        return np.sort(self.dynamical.gamma_frequencies())


class PhononPipeline:
    def __init__(self, primitive, supercell_scale=2, backend="analytic",
                 backend_kwargs=None, executor=None, distance=0.01,
                 symprec=1e-5, device="auto", verbose=True):
        self.primitive = primitive
        self.scale = supercell_scale
        self.backend = get_backend(backend, **(backend_kwargs or {}))
        self.executor = executor
        self.distance = distance
        self.symprec = symprec
        self.device = device
        self.verbose = verbose

    def run(self):
        prim = self.primitive
        sc, cmap, pmap = prim.make_supercell_with_map(self.scale)
        plan = displacement_plan(sc, distance=self.distance, symprec=self.symprec)
        evals = flatten_evals(plan)
        n_evals = len(evals)
        n_full = 6 * sc.natoms
        if self.verbose:
            print(f"[pipeline] primitive atoms={prim.natoms}  supercell atoms={sc.natoms}")
            print(f"[pipeline] irreducible displacement atoms={len(plan)}")
            print(f"[pipeline] force evaluations={n_evals}  (naive full = {n_full}; "
                  f"{n_full/max(n_evals,1):.1f}x fewer via symmetry)")
            if isinstance(self.backend, type(self.backend)):
                print(f"[pipeline] backend='{self.backend.name}'  executor={self.executor}")
        forces = self.backend.evaluate(sc, evals, self.executor)
        fc = assemble_from_forces(sc, plan, forces, symprec=self.symprec)
        dyn = DynamicalMatrix(prim, sc, fc, cmap, pmap, symprec=self.symprec)
        return PhononResult(prim, sc, fc, dyn)
