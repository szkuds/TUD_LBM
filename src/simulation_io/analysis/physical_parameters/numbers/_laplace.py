"""Laplace number: surface tension and inertia against viscous dissipation."""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.registry import dimensionless_operator

if TYPE_CHECKING:
    from src.simulation_io.analysis.physical_parameters._inputs import DimensionlessInputs


@dimensionless_operator(
    name="la",
    label=r"$\mathrm{La}$",
    row_label="La (Laplace number):",
    formula="γ*L*ρ_l/ν² = 1/Oh²",
    order=20,
    needs_gravity=False,
)
def laplace_number(inputs: DimensionlessInputs) -> float | None:
    """La = γ·L·ρ_l / ν², the reciprocal square of :func:`ohnesorge_number`.

    Written out rather than as ``1/oh**2``: the direct form stays finite as
    ``Oh -> 0`` and keeps the two numbers independent of each other, while
    building both from the same :class:`DimensionlessInputs` is what guarantees
    the identity holds. Because it *is* an exact reparametrisation of ``Oh``, it
    is excluded from automatic legend labelling — see
    :mod:`src.simulation_io.plotting.run_labels`.

    The numerator guard mirrors :func:`ohnesorge_number`'s denominator guard so
    the identity holds at the edges too: without it a run with ``gamma == 0``
    would leave ``Oh`` unresolved while reporting ``La = 0.0``.
    """
    numerator = inputs.gamma * inputs.length * inputs.rho_l
    if numerator <= 0.0 or inputs.nu <= 0.0:
        return None
    return numerator / inputs.nu**2
