from __future__ import annotations

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

_TWO_PI = 2.0 * np.pi


def _torch_available():
    return _HAS_TORCH


class DynamicalMatrix:
    def __init__(self, prim, superstruct, fc_sc, cell_map, prim_map, symprec=1e-5):
        self.prim = prim
        self.Np = prim.natoms
        self.Nc = superstruct.natoms // prim.natoms
        n3_sc = fc_sc.shape[0]
        n3_p = 3 * self.Np
        sc_index_of = {}
        for idx, (cell, p) in enumerate(zip(cell_map, prim_map)):
            sc_index_of[(tuple(cell), p)] = idx
        origin_cell = (0, 0, 0)
        origin_sc = {kp: sc_index_of[(origin_cell, kp)] for kp in range(self.Np)}
        cells = sorted({tuple(c) for c in cell_map})
        self.cells = cells
        Ncell = len(cells)
        blocks = np.zeros((self.Np, self.Np, Ncell, 3, 3), dtype=float)
        tvec = np.zeros((self.Np, self.Np, Ncell, 3), dtype=float)
        for ka in range(self.Np):
            i0 = origin_sc[ka]
            for kb in range(self.Np):
                for ridx, cell in enumerate(cells):
                    j = sc_index_of.get((cell, kb))
                    if j is None:
                        continue
                    blocks[ka, kb, ridx] = fc_sc[3 * i0:3 * i0 + 3, 3 * j:3 * j + 3]
                    tv = np.array(cell, dtype=float) + prim.positions[kb] - prim.positions[ka]
                    tvec[ka, kb, ridx] = tv
        self.blocks = blocks
        self.tvec = tvec
        mk = prim.masses
        invsqrt = 1.0 / np.sqrt(np.outer(mk, mk))
        self.mass_weight = np.broadcast_to(invsqrt[:, :, None, None, None], blocks.shape)

    def to(self, device, dtype=None):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch not available; cannot move to device")
        self._blocks_t = torch.tensor(self.blocks, device=device, dtype=torch.complex128)
        self._mw_t = torch.tensor(self.mass_weight, device=device, dtype=torch.complex128)
        self._tvec_t = torch.tensor(self.tvec, device=device, dtype=torch.float64)
        self._device = device
        return self

    def build_batch(self, q_frac, device="auto"):
        q = np.asarray(q_frac, dtype=float).reshape(-1, 3)
        if device in ("auto", None) and _HAS_TORCH and torch.cuda.is_available():
            return self._build_torch(q, "cuda")
        if _HAS_TORCH and isinstance(device, str) and device.startswith("cuda"):
            return self._build_torch(q, device)
        return self._build_numpy(q)

    def _build_numpy(self, q):
        Np = self.Np
        phase = np.exp(1j * _TWO_PI * np.einsum("qc,abrc->qabr", q, self.tvec))
        D = np.einsum("abrde,qabr->qabde", self.blocks * self.mass_weight, phase)
        nq = q.shape[0]
        Dmat = np.zeros((nq, 3 * Np, 3 * Np), dtype=complex)
        for ka in range(Np):
            for kb in range(Np):
                Dmat[:, 3 * ka:3 * ka + 3, 3 * kb:3 * kb + 3] = D[:, ka, kb, :, :]
        return 0.5 * (Dmat + np.conj(np.transpose(Dmat, (0, 2, 1))))

    def _build_torch(self, q, device):
        if not hasattr(self, "_blocks_t") or getattr(self, "_device", None) != device:
            self.to(device)
        qt = torch.tensor(q, device=device, dtype=self._tvec_t.dtype)
        phase = torch.exp((_TWO_PI * 1j) * torch.einsum("qc,abrc->qabr", qt, self._tvec_t))
        weighted = self._blocks_t * self._mw_t
        D = torch.einsum("abrde,qabr->qabde", weighted, phase)
        nq = q.shape[0]
        Np = self.Np
        Dmat = torch.zeros((nq, 3 * Np, 3 * Np), dtype=weighted.dtype, device=device)
        for ka in range(Np):
            for kb in range(Np):
                Dmat[:, 3 * ka:3 * ka + 3, 3 * kb:3 * kb + 3] = D[:, ka, kb, :, :]
        Dh = 0.5 * (Dmat + Dmat.conj().transpose(1, 2))
        return Dh

    def frequencies(self, q_frac, device="auto"):
        q = np.asarray(q_frac, dtype=float).reshape(-1, 3)
        if _HAS_TORCH and (device == "auto" and torch.cuda.is_available() or
                           (isinstance(device, str) and device.startswith("cuda"))):
            dev = "cuda" if device == "auto" else device
            D = self._build_torch(q, dev)
            try:
                eig = torch.linalg.eigvalsh(D)
                freqs = torch.sign(eig) * torch.sqrt(torch.abs(eig))
                return freqs.cpu().numpy()
            except Exception:
                pass
        D = self._build_numpy(q)
        eig = np.linalg.eigvalsh(D)
        return np.sign(eig) * np.sqrt(np.abs(eig))

    def gamma_frequencies(self, device="auto"):
        return self.frequencies(np.zeros((1, 3)), device=device)[0]

    def frequencies_multi_gpu(self, q_frac, devices=None):
        if not _HAS_TORCH or not torch.cuda.is_available():
            return self.frequencies(q_frac, device="cpu")
        if devices is None:
            devices = list(range(torch.cuda.device_count()))
        q = np.asarray(q_frac, dtype=float).reshape(-1, 3)
        ndev = len(devices)
        chunks = [q[i::ndev] for i in range(ndev)]
        out = [None] * q.shape[0]
        for ci, dev in enumerate(devices):
            if chunks[ci].shape[0] == 0:
                continue
            D = self._build_torch(chunks[ci], f"cuda:{dev}")
            eig = torch.linalg.eigvalsh(D)
            freqs = (torch.sign(eig) * torch.sqrt(torch.abs(eig))).cpu().numpy()
            for k, gi in enumerate(range(ci, q.shape[0], ndev)):
                out[gi] = freqs[k]
        return np.array(out)
