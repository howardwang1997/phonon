"""phonon_accel — GPU-accelerated, MLIP-and-DFT phonon pipeline.

Two research lines share one phonon engine (``phonons.PhononCalculation``):

* Line A — benchmark + fine-tune foundation MLIPs for phonon accuracy.
* Line B — accelerate the DFT finite-displacement workflow on GPU.

The engine is backend-agnostic: any ASE calculator (an MLIP from
``mlip_calc`` or a DFT backend) plugs into the same
displacements -> forces -> force-constants -> bands/DOS/thermal flow.
"""

__version__ = "0.1.0"

from .phonons import PhononCalculation  # noqa: F401
from .long_range import (  # noqa: F401
    FewQChebyshevVertexAdapter,
    ProjectedLongRangeResult,
    apply_mode_projected_correction,
)
