from .base import ForceBackend
from .analytic import AnalyticForceBackend
from .qe_gpu import QEGPUBackend
from .vasp_gpu import VASPGPUBackend


def get_backend(name, **kwargs):
    name = name.lower()
    if name in ("analytic", "spring"):
        return AnalyticForceBackend(**kwargs)
    if name in ("qe", "qe-gpu", "quantumespresso"):
        return QEGPUBackend(**kwargs)
    if name in ("vasp", "vasp-gpu"):
        return VASPGPUBackend(**kwargs)
    raise ValueError(f"unknown backend '{name}'; choose analytic|qe-gpu|vasp-gpu")


__all__ = ["ForceBackend", "AnalyticForceBackend", "QEGPUBackend", "VASPGPUBackend", "get_backend"]
