"""Line A3 — fine-tuning foundation MLIPs to fix phonon failure modes.

Core idea: distill the genuine DFPT second-order force constants (Φ) from the
MDR database into the foundation model, using *zero new DFT*. For each training
material we generate rattled supercells and label them with the exact harmonic
forces  F = -Φ·u  and energy  E = ½ uᵀΦu. Force-matching on these configs
teaches the model the correct curvature (Hessian) → removes the systematic
softening and spurious imaginary modes.
"""

from .dataset import (  # noqa: F401
    harmonic_energy,
    harmonic_forces,
    make_training_configs,
)
