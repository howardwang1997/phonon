from __future__ import annotations

import argparse
import time

import numpy as np

from .structure import Structure
from .force_constants import enforce_acoustic_sum_rule
from .dynamical import DynamicalMatrix, _torch_available

if _torch_available():
    import torch


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


def _make_system(n_prim_atoms, sc_size, a=5.43):
    fcc = a * np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    reps = max(1, int(np.ceil(n_prim_atoms / 8)))
    lattice = a * reps * np.eye(3)
    basis = []
    for ix in range(reps):
        for iy in range(reps):
            for iz in range(reps):
                for shift in (fcc, fcc + 0.25 * a):
                    for p in shift:
                        frac = (p / a + np.array([ix, iy, iz])) / reps
                        basis.append(frac)
    prim = Structure(lattice, basis, ["Si"] * len(basis))
    sc, cmap, pmap = prim.make_supercell_with_map([sc_size, sc_size, sc_size])
    return prim, sc, cmap, pmap, a


def _time_solver(dyn, qgrid, device, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        dyn.frequencies(qgrid, device=device)
        if device != "cpu" and _torch_available() and torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description="CPU-vs-GPU phonon-solver benchmark")
    ap.add_argument("--nq", type=int, default=8000, help="number of q-points (e.g. 20^3=8000)")
    ap.add_argument("--sc", type=int, default=3, help="supercell factor")
    ap.add_argument("--prim-reps", type=int, default=2, help="primitive size in units of 8 atoms")
    args = ap.parse_args(argv)

    nprim = args.prim_reps * 8
    prim, sc, cmap, pmap, a = _make_system(nprim, args.sc)
    H = _spring_fc(sc, 1.0, np.sqrt(3) / 4 * a * 1.05)
    dyn = DynamicalMatrix(prim, sc, H, cmap, pmap)

    n = int(round(args.nq ** (1 / 3)))
    xs = np.linspace(0, 1, n, endpoint=False)
    qgrid = np.array(np.meshgrid(xs, xs, xs, indexing="ij")).reshape(3, -1).T
    nq = qgrid.shape[0]
    nbranches = 3 * prim.natoms

    print(f"primitive atoms : {prim.natoms}  (phonon branches: {nbranches})")
    print(f"supercell atoms : {sc.natoms}  (force-constant matrix: {3*sc.natoms}x{3*sc.natoms})")
    print(f"q-points        : {nq}  ({n}^3 grid)")
    print(f"dynamical matrices per solve : {nq} x {nbranches}x{nbranches} (complex Hermitian)")
    print("-" * 60)

    t_cpu = _time_solver(dyn, qgrid, "cpu")
    print(f"CPU (numpy, single-thread loop) : {t_cpu*1000:10.1f} ms")
    print("-" * 60)

    if _torch_available() and torch.cuda.is_available():
        ndev = torch.cuda.device_count()
        single_gpu_times = []
        for d in range(ndev):
            name = torch.cuda.get_device_name(d)
            t_gpu = _time_solver(dyn, qgrid, f"cuda:{d}")
            single_gpu_times.append(t_gpu)
            print(f"GPU cuda:{d} ({name:16s})  : {t_gpu*1000:10.1f} ms   speedup x{t_cpu/t_gpu:6.1f}")
        print("-" * 60)
        t_best = min(single_gpu_times)

        def _time_multi(repeats=3):
            best = float("inf")
            for _ in range(repeats):
                t0 = time.perf_counter()
                dyn.frequencies_multi_gpu(qgrid)
                torch.cuda.synchronize()
                best = min(best, time.perf_counter() - t0)
            return best

        if ndev > 1:
            t_multi = _time_multi()
            tag = "faster" if t_multi < t_best else "no gain (problem too small; multi-GPU is for SCF pipeline)"
            print(f"GPU x{ndev} (q-grid split)        : {t_multi*1000:10.1f} ms   [{tag}]")
            print("-" * 60)
            print(f"best single-GPU speedup         : x{t_cpu/t_best:6.1f}")
        else:
            print(f"best GPU vs CPU speedup         : x{t_cpu/t_best:6.1f}")
    else:
        print("No CUDA available; cannot benchmark GPU.")


if __name__ == "__main__":
    main()
