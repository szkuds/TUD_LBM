"""Double-well EOS: bulk chemical potential, bulk pressure and closed-form surface tension."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from src.registry import eos_operator
from src.registry import pressure_operator
from src.registry import surface_tension_operator

if TYPE_CHECKING:
    import jax.numpy as jnp
    from src.operators.macroscopic import MultiphaseParams
    from src.operators.protocols import EOSFunction
    from src.operators.protocols import PressureFunction


def _beta(mp: MultiphaseParams) -> float:
    """Well depth implied by the interface width and the density contrast.

    Shared by both builders so the chemical potential and the pressure can
    never be derived from different well depths.
    """
    return 8.0 * mp.kappa / (float(mp.interface_width) ** 2 * (mp.rho_l - mp.rho_v) ** 2)


def _eos_double_well(
    rho: jnp.ndarray,
    beta: float,
    rho_l: float,
    rho_v: float,
) -> jnp.ndarray:
    """Double-well equation-of-state derivative for bulk chemical potential."""
    return 2.0 * beta * (rho - rho_l) * (rho - rho_v) * (2.0 * rho - rho_l - rho_v)


def _pressure_double_well(
    rho: jnp.ndarray | np.ndarray,
    beta: float,
    rho_l: float,
    rho_v: float,
) -> jnp.ndarray | np.ndarray:
    """Double-well bulk thermodynamic pressure ``p_0(rho)``.

    ``p_0 = beta * (rho - rho_l) * (rho - rho_v) * (3.0 * rho ** 2 - rho_l*rho_v - rho * (rho_l+rho_v))``
    with the bulk free-energy density ``psi(rho) = beta * (rho - rho_l)^2 * (rho - rho_v)^2``, so it is
    exactly consistent with ``_eos_double_well`` (``mu_0 = d(psi)/d(rho)``). Used by the
    surface-tension calibration and the pressure plots; not part of the force pipeline.
    Plain arithmetic, so it accepts NumPy or JAX arrays.
    """
    return beta * (rho - rho_l) * (rho - rho_v) * (3.0 * rho**2 - rho_l * rho_v - rho * (rho_l + rho_v))


@eos_operator(name="double-well")
def build_double_well_eos(mp: MultiphaseParams) -> EOSFunction:
    """Return ``eos_fn(rho)`` for the double-well EOS using bound params."""
    beta = _beta(mp)
    return lambda rho: _eos_double_well(rho, beta, mp.rho_l, mp.rho_v)


@pressure_operator(name="double-well")
def build_double_well_pressure(mp: MultiphaseParams) -> PressureFunction:
    """Return ``pressure_fn(rho)`` for the double-well bulk pressure using bound params."""
    beta = _beta(mp)
    return lambda rho: np.asarray(_pressure_double_well(rho, beta, mp.rho_l, mp.rho_v))


@surface_tension_operator(name="double-well")
def double_well_surface_tension(mp: MultiphaseParams) -> float | None:
    """Closed-form liquid-gas surface tension ``gamma = (2/3)(kappa/W)(drho)^2``.

    The planar-interface integral of the double-well free energy, so it is the
    same free energy :func:`_eos_double_well` and :func:`_pressure_double_well`
    differentiate (equivalently ``beta * W * drho**4 / 12`` via :func:`_beta`).

    Registering it is what tells the analysis layer that this EOS does *not*
    need a Young-Laplace calibration; an EOS with no closed form simply does
    not register here. Returns ``None`` when the interface width cannot define
    an interface, so the caller falls back to a measured value rather than
    dividing by zero.
    """
    if mp.interface_width <= 0:
        return None
    return (2.0 / 3.0) * (float(mp.kappa) / float(mp.interface_width)) * (float(mp.rho_l) - float(mp.rho_v)) ** 2
