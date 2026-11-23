"""Kelvin equation: the shift of the vapour pressure at a curved interface, relative to the flat one.

The vapour pressure at a curved liquid-vapour interface differs from the
saturation pressure ``P`` at a flat one (W. Thomson, Phil. Mag. 42, 448, 1871; in
the general form given at https://en.wikipedia.org/wiki/Kelvin_equation)::

    p(r_1, r_2) = P - gamma * rho_v / (rho_l - rho_v) * (1/r_1 + 1/r_2)

for a concave interface. With the Ostwald-Freundlich sign convention of the same
article — ``r > 0`` for a convex surface (a droplet), ``r < 0`` for a concave one
(a bubble) — and in two dimensions, where the interface is a cylinder and only
one principal radius is finite, this module reports the relative shift::

    (p(r) - P) / P = gamma * rho_v / ((rho_l - rho_v) * P * r)

It is positive for a droplet (the vapour around it is above saturation) and
negative for a bubble (the vapour inside it is below). ``P`` is the EOS's own
saturation pressure, ``p_0`` at the coexistence vapour density.

Kelvin's form is the first order in ``1/r``. The Ostwald-Freundlich form
``ln(p/P) = gamma * V_m / (r * R * T)`` (again with one radius in 2D) adds an
ideal vapour and an incompressible liquid. Neither assumption holds here, so the
densities themselves are solved for, without either, by
:func:`~src.operators.macroscopic.eos.kelvin_densities`, and
``physical_parameters.txt`` prints that pair beside this number.

Why it matters in a diffuse-interface model: the shift scales with the interface
width over the radius. In a real fluid it is negligible for any visible bubble.
The 2026-10-17 bubbles, a few interface widths across, sat at -0.34 to -0.50:
their vapour relaxed to 0.0103-0.0118 against a saturation density of 0.0168, a
density ratio of 1000-1200 inside the bubble, and every one that moved lost that
vapour to a two-cell mode. Only a larger radius relative to the interface width,
or a higher reduced temperature, makes the shift smaller.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from src.registry import dimensionless_operator

if TYPE_CHECKING:
    from src.simulation_io.analysis.physical_parameters._inputs import DimensionlessInputs


@dimensionless_operator(
    name="kelvin_shift",
    label=r"$(p_r - p_\mathrm{sat})/p_\mathrm{sat}$",
    row_label="Kelvin shift (p_r-P)/P:",
    formula="(p(r)-P)/P = γ*ρ_v/((ρ_l-ρ_v)*P*r), r<0 concave (bubble)",
    order=90,
    needs_gravity=False,
    # Built on the radius of curvature and the coexistence pair, not on the
    # inclusion length or the buoyancy contrast.
    annotates_scale=False,
)
def kelvin_pressure_shift(inputs: DimensionlessInputs) -> float | None:
    """``(p(r) - P) / P`` by the Kelvin equation for the inclusion's interface.

    ``None`` without a signed radius of curvature, without a coexistence pair,
    without an EOS registered under the ``pressure`` kind, or when the EOS has no
    positive saturation pressure to compare with (the double-well: ``p_0`` is zero
    at both wells).
    """
    if inputs.curvature_radius is None or inputs.coexistence is None or inputs.pressure is None:
        return None
    rho_v, rho_l = inputs.coexistence
    p_sat = float(np.asarray(inputs.pressure(np.asarray([rho_v])), dtype=float)[0])
    denominator = (rho_l - rho_v) * p_sat
    if denominator <= 0.0 or abs(inputs.curvature_radius) <= 0.0:
        return None
    return inputs.gamma * rho_v / (denominator * inputs.curvature_radius)
