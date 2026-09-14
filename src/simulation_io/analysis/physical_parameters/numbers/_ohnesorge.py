"""Ohnesorge number: viscous forces against inertia and surface tension."""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.registry import dimensionless_operator

if TYPE_CHECKING:
    from src.simulation_io.analysis.physical_parameters._inputs import DimensionlessInputs


@dimensionless_operator(
    name="oh",
    label=r"$\mathrm{Oh}$",
    row_label="Oh (Ohnesorge number):",
    formula="ν/sqrt(ρ_l*γ*L)",
    order=10,
    needs_gravity=False,
)
def ohnesorge_number(inputs: DimensionlessInputs) -> float | None:
    """Oh = ν / sqrt(γ·L·ρ_l)."""
    denominator = inputs.gamma * inputs.length * inputs.rho_l
    if denominator <= 0.0:
        return None
    return inputs.nu / denominator**0.5
