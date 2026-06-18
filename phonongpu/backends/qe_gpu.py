from __future__ import annotations

import os
import re
import shutil
import subprocess
import numpy as np

from .base import ForceBackend
from ..structure import Structure


class QEGPUBackend(ForceBackend):
    name = "qe-gpu"

    def __init__(self, pseudo_dir=".", pseudos=None, ecutwfc=50.0, ecutrho=400.0,
                 kgrid=(4, 4, 4), pw_exe="pw.x", ncpu=1, options=None):
        super().__init__(options)
        self.pseudo_dir = pseudo_dir
        self.pseudos = pseudos or {}
        self.ecutwfc = ecutwfc
        self.ecutrho = ecutrho
        self.kgrid = tuple(kgrid)
        self.pw_exe = pw_exe
        self.ncpu = ncpu

    def _pseudos_for(self, structure):
        out = []
        for sp in dict.fromkeys(structure.species):
            if sp in self.pseudos:
                out.append(self.pseudos[sp])
            else:
                out.append(f"{sp}.UPF")
        return out

    def write_input(self, structure, path, title="scf"):
        species = list(dict.fromkeys(structure.species))
        pseudo_list = self._pseudos_for(structure)
        masses = {s: Structure(np.eye(3), [[0, 0, 0]], [s]).masses[0] for s in species}
        lines = []
        lines.append("&CONTROL")
        lines.append(f"  prefix='{title}', pseudo_dir='{self.pseudo_dir}',")
        lines.append("  calculation='scf', tprnfor=.true.,")
        lines.append("/")
        lines.append("&SYSTEM")
        lines.append(f"  ibrav=0, nat={structure.natoms}, ntyp={len(species)},")
        lines.append(f"  ecutwfc={self.ecutwfc}, ecutrho={self.ecutrho},")
        lines.append("  input_dft='pbe',")
        lines.append("/")
        lines.append("&ELECTRONS")
        lines.append("  conv_thr=1.0d-8,")
        lines.append("/")
        lines.append("ATOMIC_SPECIES")
        for s, p in zip(species, pseudo_list):
            lines.append(f"  {s} {masses[s]:.5f} {p}")
        lines.append("CELL_PARAMETERS angstrom")
        for row in structure.cell:
            lines.append(f"  {row[0]: .12f} {row[1]: .12f} {row[2]: .12f}")
        lines.append("ATOMIC_POSITIONS crystal")
        for s, p in zip(structure.species, structure.positions):
            lines.append(f"  {s} {p[0]: .12f} {p[1]: .12f} {p[2]: .12f}")
        lines.append("K_POINTS automatic")
        lines.append(f"  {self.kgrid[0]} {self.kgrid[1]} {self.kgrid[2]}  0 0 0")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    def parse_forces(self, out_path):
        forces = []
        pat = re.compile(
            r"atom\s+\d+\s+type\s+\d+\s+force\s*=\s*(-?[\d.Ee+\-]+)\s+"
            r"(-?[\d.Ee+\-]+)\s+(-?[\d.Ee+\-]+)"
        )
        last = None
        with open(out_path) as fh:
            for line in fh:
                m = pat.search(line)
                if m:
                    last = [float(m.group(i)) for i in (1, 2, 3)]
                    forces.append(last)
        if not forces:
            raise RuntimeError(f"no forces found in {out_path}")
        n = len(forces)
        nat = None
        with open(out_path) as fh:
            for line in fh:
                if "number of atoms/cell" in line:
                    nat = int(line.split("=")[-1].strip())
                    break
        if nat and n >= nat:
            forces = forces[-nat:]
        return np.array(forces, dtype=float)

    def run_one(self, workdir, gpu_id, dry_run=False):
        inp = os.path.join(workdir, "scf.in")
        out = os.path.join(workdir, "scf.out")
        if dry_run or not shutil.which(self.pw_exe):
            return out
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        use_mpi = self.options.get("use_mpi", True)
        if use_mpi:
            mpi = self.options.get("mpirun_exe", "mpirun")
            args = self.options.get("mpirun_args", ["--allow-run-as-root", "-np", "1"])
            cmd = [mpi] + args + [self.pw_exe, "-i", inp]
        else:
            cmd = [self.pw_exe, "-nk", str(max(1, self.ncpu)), "-i", inp]
        with open(out, "w") as fout:
            subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT,
                           env=env, cwd=workdir, check=True)
        return out

    def evaluate(self, structure, evals, executor=None):
        N = structure.natoms
        root = self.options.get("workdir", "./phonongpu_qe")
        gpu_ids = self.options.get("gpu_ids")
        if gpu_ids is None:
            gpu_ids = list(range(max(self.options.get("num_gpus", 1), 1)))
        os.makedirs(root, exist_ok=True)
        jobs = []
        for ie, (idx, vec) in enumerate(evals):
            disp = structure.copy()
            disp.positions[idx] = disp.positions[idx] + np.linalg.inv(disp.cell) @ np.asarray(vec)
            disp.positions[idx] -= np.floor(disp.positions[idx])
            wdir = os.path.join(root, f"disp_{ie:05d}")
            os.makedirs(wdir, exist_ok=True)
            self.write_input(disp, os.path.join(wdir, "scf.in"))
            jobs.append((ie, wdir))

        def _work(args):
            ie, wdir, gpu = args
            dry = self.options.get("dry_run", False)
            out = self.run_one(wdir, gpu, dry_run=dry)
            if dry or not shutil.which(self.pw_exe):
                return ie, np.zeros((N, 3))
            return ie, self.parse_forces(out)

        gpu_assign = [gpu_ids[i % len(gpu_ids)] for i in range(len(jobs))]
        if executor is not None:
            results = executor.map(_work, [(ie, wdir, g) for (ie, wdir), g in zip(jobs, gpu_assign)])
        else:
            results = [_work((ie, wdir, g)) for (ie, wdir), g in zip(jobs, gpu_assign)]
        forces = np.zeros((len(evals), N, 3))
        for ie, f in results:
            forces[ie] = f
        return forces


def _cycle_gpus(n, total):
    return [i % max(n, 1) for i in range(total)]
