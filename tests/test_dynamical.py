import numpy as np
import pytest

from phonongpu.structure import Structure
from phonongpu.force_constants import enforce_acoustic_sum_rule
from phonongpu.dynamical import DynamicalMatrix

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except Exception:
    HAS_TORCH = False


def _diamond_system(sc=3, a=5.43):
    prim_lattice = a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    prim = Structure(prim_lattice, [[0, 0, 0], [0.25, 0.25, 0.25]], ["Si", "Si"])
    sc_st, cmap, pmap = prim.make_supercell_with_map([sc, sc, sc])
    return prim, sc_st, cmap, pmap, a


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


def _build_dyn(sc=3):
    prim, sc_st, cmap, pmap, a = _diamond_system(sc)
    H = _spring_fc(sc_st, 1.0, np.sqrt(3) / 4 * a * 1.05)
    return DynamicalMatrix(prim, sc_st, H, cmap, pmap), prim


def test_gamma_acoustic_zeros():
    dyn, prim = _build_dyn(sc=2)
    g = np.sort(dyn.gamma_frequencies())
    assert g.shape[0] == 3 * prim.natoms
    assert np.max(np.abs(g[:3])) < 1e-6
    assert np.all(g[3:] > -1e-3)


def test_stability_along_path():
    dyn, _ = _build_dyn(sc=2)
    qs = np.linspace([0, 0, 0], [0.45, 0.45, 0.45], 15)
    fr = dyn.frequencies(qs)
    assert (fr > -1e-3).all()


@pytest.mark.skipif(not HAS_TORCH, reason="CUDA not available")
def test_gpu_cpu_agreement():
    dyn, _ = _build_dyn(sc=2)
    qs = np.linspace([0, 0, 0], [0.4, 0.4, 0.4], 20)
    gpu = np.sort(dyn.frequencies(qs, device="cuda:0"), axis=1)
    cpu = np.sort(dyn.frequencies(qs, device="cpu"), axis=1)
    assert np.abs(gpu - cpu).max() < 1e-6
