"""Bond number and its components along/across an inclined gravity vector."""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
from typing import NamedTuple
from src.registry import dimensionless_operator

if TYPE_CHECKING:
    from src.simulation_io.analysis.physical_parameters._inputs import DimensionlessInputs


class BondNumbers(NamedTuple):
    """Bond number and its components along/across the inclined gravity vector."""

    bo: float
    bo_perp: float
    bo_parallel: float


def compute_bond_numbers(
    delta_rho_phases: float,
    gamma: float,
    g_val: float,
    length: float,
    angle_deg: float = 0.0,
) -> BondNumbers:
    """Bo = (Δρ*g*L²)/γ, split into normal/tangential components by angle_deg."""
    bo = (delta_rho_phases * (length**2) * g_val) / gamma
    angle_rad = math.radians(angle_deg)
    return BondNumbers(bo=bo, bo_perp=bo * math.cos(angle_rad), bo_parallel=bo * math.sin(angle_rad))


def _bond(inputs: DimensionlessInputs) -> BondNumbers | None:
    """The whole family for one run, or ``None`` without gravity.

    All three operators go through here so the decomposition can never drift
    apart from the total it decomposes.
    """
    if inputs.g is None or inputs.gamma == 0.0:
        return None
    return compute_bond_numbers(inputs.drho, inputs.gamma, inputs.g, inputs.length, inputs.angle_deg or 0.0)


@dimensionless_operator(
    name="bo",
    label=r"$\mathrm{Bo}$",
    row_label="Bo (Bond number):",
    formula="(ΔρgL²)/γ",
    order=30,
    needs_gravity=True,
)
def bond_number(inputs: DimensionlessInputs) -> float | None:
    """Bo = (Δρ·g·L²)/γ."""
    numbers = _bond(inputs)
    return None if numbers is None else numbers.bo


@dimensionless_operator(
    name="bo_perp",
    label=r"$\mathrm{Bo}_{\perp}$",
    row_label="Bo_perp (Bond normal):",
    formula="(Δρ*g*cos(θ)*L²)/γ",
    order=40,
    needs_gravity=True,
)
def bond_number_normal(inputs: DimensionlessInputs) -> float | None:
    """The component of Bo normal to an inclined wall."""
    numbers = _bond(inputs)
    return None if numbers is None else numbers.bo_perp


@dimensionless_operator(
    name="bo_parallel",
    label=r"$\mathrm{Bo}_{\parallel}$",
    row_label="Bo_parallel (Bond tangential):",
    formula="(Δρ*g*sin(θ)*L²)/γ",
    order=50,
    needs_gravity=True,
)
def bond_number_tangential(inputs: DimensionlessInputs) -> float | None:
    """The component of Bo tangential to an inclined wall -- what drives motion."""
    numbers = _bond(inputs)
    return None if numbers is None else numbers.bo_parallel
