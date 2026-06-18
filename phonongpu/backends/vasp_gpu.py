from __future__ import annotations

import os
import re
import shutil
import subprocess
import numpy as np

from .base import ForceBackend
from ..structure import Structure
from ..io.poscar import write_poscar


class VASPGPUBackend(ForceBackend):
    name = "vasp-gpu"

    def __init__(self, incar=None, kpoints=None, potcar_dir=".",
                 vasp_exe="vasp_std", options=None):
        super().__init__(options)
        self.incar = incar or {}
        self.kpoints = kpoints or (8, 8, 8)
        self.potcar_dir = potcar_dir
        self.vasp_exe = vasp_exe

    def write_incar(self, path):
        defaults = {
            "ENCUT": 520, "ISMEAR": 0, "SIGMA": 0.05, "EDIFF": 1e-8,
            "IBRION": -1, "NSW": 0, "LREAL": ".FALSE.", "LWAVE": ".FALSE.",
            "LCHARG": ".FALSE.", "PREC": "Accurate", "ICHARG": 2,
        }
        defaults.update(self.incar)
        lines = [f"{k} = {v}" for k, v in defaults.items()]
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    def write_kpoints(self, path):
        k = self.kpoints
        with open(path, "w") as fh:
            fh.write("K-Points\n0\nGamma\n")
            if hasattr(k, "__len__") and len(k) == 3:
                fh.write(f"{k[0]} {k[1]} {k[2]}\n0 0 0\n")
            else:
                fh.write(f"{k} {k} {k}\n0 0 0\n")

    def write_potcar(self, structure, path):
        unique = list(dict.fromkeys(structure.species))
        parts = []
        for sp in unique:
            p = os.path.join(self.potcar_dir, f"POTCAR.{sp}")
            if os.path.exists(p):
                with open(p) as fh:
                    parts.append(fh.read())
            else:
                parts.append(f"# missing POTCAR for {sp}\n")
        with open(path, "w") as fh:
            fh.write("".join(parts))

    def parse_forces(self, outcar):
        forces = []
        with open(outcar) as fh:
            inblock = False
            for line in fh:
                if "TOTAL-FORCE" in line:
                    inblock = True
                    forces = []
                    continue
                if inblock:
                    parts = line.split()
                    if len(parts) >= 6 and re.match(r"^\s*-?\d", line):
                        forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
                    elif "------" in line and forces:
                        inblock = False
        if not forces:
            raise RuntimeError(f"no forces in {outcar}")
        return np.array(forces, dtype=float)

    def run_one(self, workdir, gpu_id, dry_run=False):
        if dry_run or not shutil.which(self.vasp_exe):
            return os.path.join(workdir, "OUTCAR")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        with open(os.path.join(workdir, "stdout.txt"), "w") as fout:
            subprocess.run([self.vasp_exe], stdout=fout, stderr=subprocess.STDOUT,
                           env=env, cwd=workdir, check=True)
        return os.path.join(workdir, "OUTCAR")

    def evaluate(self, structure, evals, executor=None):
        N = structure.natoms
        root = self.options.get("workdir", "./phonongpu_vasp")
        os.makedirs(root, exist_ok=True)
        jobs = []
        for ie, (idx, vec) in enumerate(evals):
            disp = structure.copy()
            disp.positions[idx] = disp.positions[idx] + np.linalg.inv(disp.cell) @ np.asarray(vec)
            disp.positions[idx] -= np.floor(disp.positions[idx])
            wdir = os.path.join(root, f"disp_{ie:05d}")
            os.makedirs(wdir, exist_ok=True)
            write_poscar(disp, os.path.join(wdir, "POSCAR"))
            self.write_incar(os.path.join(wdir, "INCAR"))
            self.write_kpoints(os.path.join(wdir, "KPOINTS"))
            self.write_potcar(disp, os.path.join(wdir, "POTCAR"))
            jobs.append((ie, wdir))

        def _work(args):
            ie, wdir, gpu = args
            dry = self.options.get("dry_run", False)
            out = self.run_one(wdir, gpu, dry_run=dry)
            if dry or not shutil.which(self.vasp_exe):
                return ie, np.zeros((N, 3))
            return ie, self.parse_forces(out)

        if executor is not None:
            results = executor.map(_work, [(ie, wdir, g) for (ie, wdir), g
                                           in zip(jobs, _cycle(len(executor.num_gpus), len(jobs)))])
        else:
            results = [_work((ie, wdir, 0)) for ie, wdir in jobs]
        forces = np.zeros((len(evals), N, 3))
        for ie, f in results:
            forces[ie] = f
        return forces


def _cycle(n, total):
    return [i % max(n, 1) for i in range(total)]
