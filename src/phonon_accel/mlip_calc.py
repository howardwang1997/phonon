"""Unified factory for foundation-MLIP ASE calculators.

Each model is imported lazily so a missing package never breaks the others.
``get_calculator(name, device=...)`` returns a ready ASE calculator.

Supported (install on demand):
    mace      -> mace-torch        (MACE-MP-0 / MACE-MPA / MACE-OMAT)
    mattersim -> mattersim
    sevennet  -> sevenn
    orb       -> orb-models        (use the *conservative* variant for phonons)
    chgnet    -> chgnet
    m3gnet    -> matgl             (baseline)

Notes
-----
Phonons depend on the Hessian (2nd derivative of energy), so prefer
*conservative* (energy-derivative) force models. Direct-force variants are
noisy at the small displacements phonons use and tend to produce spurious
imaginary modes.
"""
from __future__ import annotations

from typing import Optional


def default_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_calculator(name: str, device: Optional[str] = None, **kwargs):
    """Return an ASE calculator for the named foundation MLIP."""
    name = name.lower()
    device = default_device(device)

    if name in ("mace", "mace-mp", "mace_mp"):
        import os

        model = kwargs.pop("model", "medium")  # small|medium|large or a path
        if isinstance(model, str) and os.path.isfile(model):
            # locally fine-tuned model file
            from mace.calculators import MACECalculator

            return MACECalculator(model_paths=[model], device=device,
                                  default_dtype="float64", **kwargs)
        from mace.calculators import mace_mp

        return mace_mp(model=model, device=device, default_dtype="float64", **kwargs)

    if name in ("mace-omat", "mace_omat"):
        from mace.calculators import mace_mp

        kwargs.pop("model", None)  # discard caller's model; OMAT uses its own
        return mace_mp(model="medium-omat-0", device=device, default_dtype="float64", **kwargs)

    if name == "mattersim":
        from mattersim.forcefield import MatterSimCalculator

        model = kwargs.pop("load_path", None) or kwargs.pop("model", None) or "MatterSim-v1.0.0-5M.pth"
        return MatterSimCalculator(load_path=model, device=device, **kwargs)

    if name in ("sevennet", "7net", "sevenn"):
        from sevenn.calculator import SevenNetCalculator

        model = kwargs.pop("model", "7net-0")
        return SevenNetCalculator(model=model, device=device, **kwargs)

    if name == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        # conservative variant recommended for phonons
        loader = kwargs.pop("loader", "orb_v3_conservative_inf_omat")
        orbff = getattr(pretrained, loader)(device=device)
        return ORBCalculator(orbff, device=device)

    if name == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator(use_device=device, **kwargs)

    if name in ("m3gnet", "matgl"):
        import matgl
        from matgl.ext.ase import PESCalculator

        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        return PESCalculator(pot, **kwargs)

    raise ValueError(f"Unknown MLIP name: {name!r}")


# Default roster for the benchmark (Line A). Comment out any not installed.
DEFAULT_MODELS = ["mace", "mattersim", "sevennet", "orb", "chgnet"]
