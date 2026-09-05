"""Temperature-conditioned conservative combination of two frozen delta MACE models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.calculators.calculator import all_changes
from ase.calculators.mixing import LinearCombinationCalculator


@dataclass(frozen=True)
class TemperatureWeights:
    temperature_K: float
    lower_temperature_K: float
    upper_temperature_K: float
    delta_300: float
    delta_600: float


class CompatibleLinearCombinationCalculator(LinearCombinationCalculator):
    """Expose the optional ASE ``system_changes`` signature used by MACE callers."""

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        return super().calculate(atoms, properties, system_changes)


def temperature_weights(
    temperature_K: float,
    *,
    lower_temperature_K: float = 300.0,
    upper_temperature_K: float = 600.0,
) -> TemperatureWeights:
    """Return the fixed linear endpoint weights inside the validated interval."""

    temperature = float(temperature_K)
    lower = float(lower_temperature_K)
    upper = float(upper_temperature_K)
    if not np.isfinite([temperature, lower, upper]).all():
        raise ValueError("temperature bounds and value must be finite")
    if not lower < upper:
        raise ValueError("lower temperature must be smaller than upper temperature")
    tolerance = 1.0e-12
    if temperature < lower - tolerance or temperature > upper + tolerance:
        raise ValueError(
            f"temperature {temperature:g} K is outside the frozen "
            f"[{lower:g}, {upper:g}] K interpolation interval"
        )
    weight_upper = (temperature - lower) / (upper - lower)
    weight_upper = float(np.clip(weight_upper, 0.0, 1.0))
    return TemperatureWeights(
        temperature_K=temperature,
        lower_temperature_K=lower,
        upper_temperature_K=upper,
        delta_300=1.0 - weight_upper,
        delta_600=weight_upper,
    )


def conditioned_short_calculator(
    base_calculator,
    delta_300_calculator,
    delta_600_calculator,
    temperature_K: float,
) -> tuple[CompatibleLinearCombinationCalculator, TemperatureWeights]:
    """Build ``E_base + (1-w) E_delta300 + w E_delta600`` at fixed temperature."""

    weights = temperature_weights(temperature_K)
    calculator = CompatibleLinearCombinationCalculator(
        [base_calculator, delta_300_calculator, delta_600_calculator],
        [1.0, weights.delta_300, weights.delta_600],
    )
    return calculator, weights
