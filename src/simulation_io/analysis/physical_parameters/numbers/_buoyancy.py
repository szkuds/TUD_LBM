"""Archimedes and Reynolds numbers for buoyancy-driven motion."""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
from src.registry import dimensionless_operator

if TYPE_CHECKING:
    from src.simulation_io.analysis.physical_parameters._inputs import DimensionlessInputs


def compute_archimedes_number(drho: float, g_val: float, length: float, nu: float, rho_l: float) -> float:
    """Ar = gL³Δρ / (ν²ρ_l)."""
    return (g_val * (length**3) * drho) / ((nu**2) * rho_l)


def compute_reynolds_number(drho: float, g_val: float, length: float, nu: float, rho_l: float) -> float:
    """Re = sqrt(Ar): characteristic buoyancy-driven Reynolds number.

    Balancing inertial drag (~ρ_l·U²·L²) against buoyancy (~Δρ·g·L³) gives the
    characteristic velocity U ~ sqrt(gLΔρ/ρ_l), so Re = UL/ν = sqrt(Ar).
    """
    ar = compute_archimedes_number(drho, g_val, length, nu, rho_l)
    return math.sqrt(ar) if ar >= 0 else math.nan


@dimensionless_operator(
    name="ar",
    label=r"$\mathrm{Ar}$",
    row_label="Ar (Archimedes number):",
    formula="gL³Δρ/(ν²ρ_l)",
    order=60,
    needs_gravity=True,
)
def archimedes_number(inputs: DimensionlessInputs) -> float | None:
    """Ar = gL³Δρ/(ν²ρ_l)."""
    if inputs.g is None or inputs.nu <= 0.0 or inputs.rho_l <= 0.0:
        return None
    return compute_archimedes_number(inputs.drho, inputs.g, inputs.length, inputs.nu, inputs.rho_l)


@dimensionless_operator(
    name="re",
    label=r"$\mathrm{Re}$",
    row_label="Re (Reynolds number):",
    formula="sqrt(Ar) = UL/ν, U=sqrt(gLΔρ/ρ_l)",
    order=70,
    needs_gravity=True,
)
def reynolds_number(inputs: DimensionlessInputs) -> float | None:
    """Re = sqrt(Ar), routed through :func:`archimedes_number`'s own formula."""
    ar = archimedes_number(inputs)
    if ar is None:
        return None
    return math.sqrt(ar) if ar >= 0 else math.nan
