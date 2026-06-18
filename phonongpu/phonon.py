from __future__ import annotations

import numpy as np

from .dynamical import DynamicalMatrix, _torch_available

if _torch_available():
    import torch

_KB = 8.617333262e-5
_THZ_PER_SQRT = 15.633302


def freqs_to_thz(freqs):
    return np.asarray(freqs) * _THZ_PER_SQRT


class PhononBands:
    def __init__(self, dyn: DynamicalMatrix):
        self.dyn = dyn

    def along_path(self, path, npoints=51, device="auto"):
        segments = []
        cum = [0.0]
        for i in range(len(path) - 1):
            a = np.array(path[i][1], dtype=float)
            b = np.array(path[i + 1][1], dtype=float)
            n = npoints if i == 0 else npoints - 1
            seg = np.linspace(a, b, n + 1)[1:] if i > 0 else np.linspace(a, b, npoints)
            segments.append(seg)
            cum.append(cum[-1] + np.linalg.norm(b - a))
        qall = np.vstack(segments)
        freqs = self.dyn.frequencies(qall, device=device)
        return qall, freqs, cum, [p[0] for p in path]


class PhononDOS:
    def __init__(self, dyn: DynamicalMatrix):
        self.dyn = dyn

    def _grid(self, mesh):
        n = int(round(mesh[0])) if hasattr(mesh, "__len__") else int(mesh)
        xs = np.linspace(0, 1, n, endpoint=False)
        g = np.array(np.meshgrid(xs, xs, xs, indexing="ij")).reshape(3, -1).T
        return g

    def dos(self, mesh=16, sigma=0.05, npts=400, freq_range=None, device="auto"):
        qgrid = self._grid(mesh)
        freqs = self.dyn.frequencies(qgrid, device=device).ravel()
        fmin = freqs.min() if freq_range is None else freq_range[0]
        fmax = freqs.max() if freq_range is None else freq_range[1]
        grid = np.linspace(fmin, fmax, npts)
        use_gpu = _torch_available() and (
            device == "auto" and torch.cuda.is_available()
            or (isinstance(device, str) and device.startswith("cuda"))
        )
        if use_gpu:
            dev = "cuda" if device == "auto" else device
            ft = torch.tensor(freqs, device=dev)
            gt = torch.tensor(grid, device=dev)
            diff = (ft[None, :] - gt[:, None]) / sigma
            dos = torch.exp(-0.5 * diff ** 2).sum(dim=1).cpu().numpy()
        else:
            diff = (freqs[None, :] - grid[:, None]) / sigma
            dos = np.exp(-0.5 * diff ** 2).sum(axis=1)
        dos /= sigma * np.sqrt(2 * np.pi)
        return grid, dos, freqs

    def thermal(self, mesh=16, tmin=0, tmax=1000, ntemp=51, device="auto"):
        qgrid = self._grid(mesh)
        freqs = np.abs(self.dyn.frequencies(qgrid, device=device))
        natoms = self.dyn.Np
        temps = np.linspace(tmin, tmax, ntemp)
        Cv = np.zeros(ntemp)
        for it, T in enumerate(temps):
            if T < 1e-6:
                continue
            x = np.maximum(freqs / T, 1e-8)
            ex = np.exp(np.clip(x, 0, 700))
            Cv[it] = (_KB * np.sum(x ** 2 * ex / (ex - 1.0) ** 2)) / natoms
        return temps, Cv
