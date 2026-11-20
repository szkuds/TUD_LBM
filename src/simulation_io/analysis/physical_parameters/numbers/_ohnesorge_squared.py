"""Squared Ohnesorge number: the viscous-capillary ratio on a linear-in-viscosity² axis."""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.registry import dimensionless_operator

if TYPE_CHECKING:
    from src.simulation_io.analysis.physical_parameters._inputs import DimensionlessInputs


@dimensionless_operator(
    name="oh2",
    label=r"$\mathrm{Oh}^2$",
    row_label="Oh² (squared Ohnesorge number):",
    formula="ν²/(ρ_l*γ*L) = 1/La",
    order=15,
    needs_gravity=False,
)
def ohnesorge_number_squared(inputs: DimensionlessInputs) -> float | None:
    """Oh² = ν² / (γ·L·ρ_l), the reciprocal of the Laplace number.

    Written out rather than as ``oh**2``, like the Laplace number: building all
    three from the same :class:`DimensionlessInputs` is what guarantees the
    identities hold, and the guard is the one :func:`ohnesorge_number` uses, so
    the two resolve for exactly the same runs. As an exact reparametrisation of
    ``Oh`` it is a regime-map axis, not new information, and is excluded from
    automatic legend labelling — see :mod:`src.simulation_io.analysis.run_labels`.
    """
    denominator = inputs.gamma * inputs.length * inputs.rho_l
    if denominator <= 0.0:
        return None
    return inputs.nu**2 / denominator
