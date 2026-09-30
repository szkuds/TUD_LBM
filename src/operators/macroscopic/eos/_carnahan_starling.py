"""Carnahan-Starling EOS: bulk chemical potential, bulk pressure and coexistence densities."""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from src.registry import eos_operator
from src.registry import pressure_operator

if TYPE_CHECKING:
    from src.config.multiphase_params import MultiphaseParams
    from src.operators.protocols import EosOperator


def _cs_params(mp: MultiphaseParams) -> tuple[float, float, float, float]:
    """Return the validated ``(a, b, r, t)`` scalars.

    Shared by both builders so the chemical potential and the pressure agree
    on which parameters are mandatory and reject the same incomplete config.
    """
    if mp.a_eos is None or mp.b_eos is None or mp.r_eos is None or mp.t_eos is None:
        msg = "a_eos, b_eos, r_eos, t_eos are all required for Carnahan-Starling EOS"
        raise ValueError(msg)
    return mp.a_eos, mp.b_eos, mp.r_eos, mp.t_eos


def _eos_carnahan_starling(
    rho: jnp.ndarray,
    a: float,
    b: float,
    r: float,
    t: float,
) -> jnp.ndarray:
    """Carnahan-Starling EOS bulk chemical potential ``mu_0(rho)``."""
    return -2.0 * a * rho + r * t * (1.0 + jnp.log(rho)) + (16.0 * r * t * (b * rho - 12.0)) / (b * rho - 4.0) ** 3


def _pressure_carnahan_starling(
    rho: jnp.ndarray,
    a: float,
    b: float,
    r: float,
    t: float,
) -> jnp.ndarray:
    """Carnahan-Starling bulk thermodynamic pressure ``p_0(rho)``.

    Consistent with ``_eos_carnahan_starling`` (the two differ only by an
    additive constant, which cancels in the Laplace pressure jump). Used by
    the surface-tension calibration and the pressure plots; not part of the
    force pipeline. Plain arithmetic, so it traces under JIT.
    """
    eta = b * rho / 4.0
    ideal = rho * r * t * (1.0 + eta + eta**2 - eta**3) / (1.0 - eta) ** 3
    return ideal - a * rho**2


@eos_operator(name="carnahan-starling")
def build_carnahan_starling_eos(mp: MultiphaseParams) -> EosOperator:
    """Return ``eos_fn(rho)`` for the Carnahan-Starling EOS using bound params."""
    a, b, r, t = _cs_params(mp)
    return lambda rho: _eos_carnahan_starling(rho, a, b, r, t)


@pressure_operator(name="carnahan-starling")
def build_carnahan_starling_pressure(mp: MultiphaseParams) -> EosOperator:
    """Return ``pressure_fn(rho) -> p_0`` for the Carnahan-Starling bulk pressure using bound params."""
    a, b, r, t = _cs_params(mp)
    return lambda rho: _pressure_carnahan_starling(rho, a, b, r, t)


# ── Coexistence densities: Maxwell equal-area construction ──────────────────
#
# Setup-time only: plain float / NumPy arithmetic rather than the JAX kernels,
# which drop to float32 when x64 is off; the construction needs double precision.

#: Relative offset from a spinodal or the ``b*rho = 4`` singularity at which a
#: bracket ends, so the root finders never evaluate a branch exactly at its limit.
_BRACKET_EPS = 1e-12

#: A subcritical isotherm turns twice; fewer sign changes means no two-phase region.
_SPINODAL_ROOT_COUNT = 2

#: Floor for the vapour-side bracket walk, so a pathological isotherm cannot spin.
_MIN_BRACKET_DENSITY = 1e-300


def _cs_pressure_float(rho: float, a: float, b: float, r: float, t: float) -> float:
    eta = b * rho / 4.0
    return rho * r * t * (1.0 + eta + eta**2 - eta**3) / (1.0 - eta) ** 3 - a * rho**2


def _cs_mu_float(rho: float, a: float, b: float, r: float, t: float) -> float:
    return -2.0 * a * rho + r * t * (1.0 + math.log(rho)) + (16.0 * r * t * (b * rho - 12.0)) / (b * rho - 4.0) ** 3


def _cs_dp_drho(rho: np.ndarray | float, a: float, b: float, r: float, t: float) -> np.ndarray | float:
    """Analytic ``dp_0/drho``: ``R*T*(Z + eta*dZ/deta) - 2*a*rho`` with ``eta = b*rho/4``.

    ``Z = (1 + eta + eta^2 - eta^3)/(1 - eta)^3`` and ``dZ/deta = (4 + 4*eta - 2*eta^2)/(1 - eta)^4``.
    """
    eta = b * rho / 4.0
    z = (1.0 + eta + eta**2 - eta**3) / (1.0 - eta) ** 3
    dz = (4.0 + 4.0 * eta - 2.0 * eta**2) / (1.0 - eta) ** 4
    return r * t * (z + eta * dz) - 2.0 * a * rho


def coexistence_densities(a: float, b: float, r: float, t: float) -> tuple[float, float]:
    """Saturated ``(rho_v, rho_l)`` at temperature *t* by Maxwell's equal-area construction.

    Solves ``p_0(rho_v) = p_0(rho_l)`` and ``mu_0(rho_v) = mu_0(rho_l)``, parametrised
    on ``rho_v`` as in the van der Waals module. Bracketing the saturation *pressure*
    between the spinodal pressures instead fails here: the liquid-branch spinodal
    pressure is negative at the usual lattice parameters, so half of that bracket
    has no vapour root.

    Raises:
        ValueError: If *t* is at or above the critical temperature (no spinodal pair).
    """
    rho_max = 4.0 / b
    grid = np.linspace(rho_max * 1e-9, rho_max * (1.0 - 1e-4), 400_001)
    sign_changes = np.flatnonzero(np.diff(np.sign(_cs_dp_drho(grid, a, b, r, t))))
    if sign_changes.size < _SPINODAL_ROOT_COUNT:
        msg = f"t = {t} is not below the Carnahan-Starling critical temperature; there is no coexistence"
        raise ValueError(msg)
    lo, hi = sign_changes[0], sign_changes[-1]
    rho_sv = brentq(_cs_dp_drho, grid[lo], grid[lo + 1], args=(a, b, r, t))
    rho_sl = brentq(_cs_dp_drho, grid[hi], grid[hi + 1], args=(a, b, r, t))
    p_sl = _cs_pressure_float(rho_sl, a, b, r, t)
    top = rho_max * (1.0 - _BRACKET_EPS)

    def liquid_at(pressure: float) -> float:
        return brentq(lambda rho: _cs_pressure_float(rho, a, b, r, t) - pressure, rho_sl, top)

    def mismatch(rho: float) -> float:
        return _cs_mu_float(liquid_at(_cs_pressure_float(rho, a, b, r, t)), a, b, r, t) - _cs_mu_float(rho, a, b, r, t)

    if p_sl > 0.0:
        # A vapour density below this has a pressure the liquid branch cannot reach.
        rho_lo = brentq(lambda rho: _cs_pressure_float(rho, a, b, r, t) - p_sl, grid[0], rho_sv)
        rho_lo += _BRACKET_EPS * (rho_sv - rho_lo)
    else:
        # mu_0 carries R*T*ln(rho), so the mismatch runs to +inf as rho -> 0; walk the
        # lower end down until it straddles the root (deep subcritical: rho_v < 1e-9 rho_max).
        rho_lo = grid[0]
        while mismatch(rho_lo) <= 0.0 and rho_lo > _MIN_BRACKET_DENSITY:
            rho_lo *= 1e-3
    rho_v = brentq(mismatch, rho_lo, rho_sv * (1.0 - _BRACKET_EPS), xtol=1e-300, rtol=1e-14)
    return float(rho_v), float(liquid_at(_cs_pressure_float(rho_v, a, b, r, t)))
