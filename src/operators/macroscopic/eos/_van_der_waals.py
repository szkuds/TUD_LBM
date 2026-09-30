r"""Van der Waals EOS: bulk chemical potential, bulk pressure and coexistence densities.

The free energy of Zhang, Guo & Wang (Phys. Fluids 34, 012110, 2022), Eqs. 4-5:

.. math::

    p_0 = \frac{\rho R T}{1 - b\rho} - a\rho^2, \qquad
    \mu_0 = R T \left[\ln\frac{\rho}{1 - b\rho} + \frac{1}{1 - b\rho}\right] - 2 a\rho

with the critical point ``rho_c = 1/(3b)``, ``T_c = 8a/(27 b R)``. The paper uses
``a = 9/392``, ``b = 2/21``, ``R = 1``, so ``rho_c = 7/2`` and ``T_c = 1/14``.
Singular at ``b*rho = 1``; the physical range ``[rho_v, rho_l]`` is well below it.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import jax.numpy as jnp
from scipy.optimize import brentq
from src.registry import eos_operator
from src.registry import pressure_operator

if TYPE_CHECKING:
    from src.config.multiphase_params import MultiphaseParams
    from src.operators.protocols import EosOperator

#: Relative offset from a spinodal or the singularity at which a bracket ends,
#: so the root finders never evaluate a branch exactly at its limit.
_BRACKET_EPS = 1e-10


def _vdw_params(mp: MultiphaseParams) -> tuple[float, float, float, float]:
    """Return the ``(a, b, r, t)`` scalars, shared by both builders.

    ``SimulationConfig._validate_multiphase`` requires all four for this EOS,
    so the asserts only narrow the optional field types.
    """
    assert mp.a_eos is not None  # noqa: S101 - guaranteed by _validate_multiphase
    assert mp.b_eos is not None  # noqa: S101 - guaranteed by _validate_multiphase
    assert mp.r_eos is not None  # noqa: S101 - guaranteed by _validate_multiphase
    assert mp.t_eos is not None  # noqa: S101 - guaranteed by _validate_multiphase
    return mp.a_eos, mp.b_eos, mp.r_eos, mp.t_eos


def _eos_van_der_waals(rho: jnp.ndarray, a: float, b: float, r: float, t: float) -> jnp.ndarray:
    """Van der Waals bulk chemical potential ``mu_0(rho)``."""
    return r * t * (jnp.log(rho / (1.0 - b * rho)) + 1.0 / (1.0 - b * rho)) - 2.0 * a * rho


def _pressure_van_der_waals(rho: jnp.ndarray, a: float, b: float, r: float, t: float) -> jnp.ndarray:
    """Van der Waals bulk pressure ``p_0(rho)``; ``dp_0/drho = rho * dmu_0/drho``."""
    return rho * r * t / (1.0 - b * rho) - a * rho**2


@eos_operator(name="van-der-waals")
def build_van_der_waals_eos(mp: MultiphaseParams) -> EosOperator:
    """Return ``eos_fn(rho)`` for the van der Waals EOS using bound params."""
    a, b, r, t = _vdw_params(mp)
    return lambda rho: _eos_van_der_waals(rho, a, b, r, t)


@pressure_operator(name="van-der-waals")
def build_van_der_waals_pressure(mp: MultiphaseParams) -> EosOperator:
    """Return ``pressure_fn(rho) -> p_0`` for the van der Waals bulk pressure using bound params."""
    a, b, r, t = _vdw_params(mp)
    return lambda rho: _pressure_van_der_waals(rho, a, b, r, t)


def coexistence_densities(a: float, b: float, r: float, t: float) -> tuple[float, float]:
    """Saturated ``(rho_v, rho_l)`` at temperature *t* by Maxwell's equal-area construction.

    Solves ``p_0(rho_v) = p_0(rho_l)`` and ``mu_0(rho_v) = mu_0(rho_l)``. Parametrised
    on ``rho_v``: each trial vapour density fixes the liquid density on the stable
    liquid branch with the same pressure, and the chemical-potential mismatch is
    bracketed between the lowest admissible vapour density and the vapour spinodal.
    Setup-time NumPy/SciPy; never traced.

    Raises:
        ValueError: If *t* is at or above the critical temperature ``8a/(27 b r)``.
    """
    t_c = 8.0 * a / (27.0 * b * r)
    if t >= t_c:
        msg = f"t = {t} is not below the critical temperature {t_c}; there is no coexistence"
        raise ValueError(msg)

    # Plain float arithmetic rather than the JAX kernels, which drop to float32
    # when x64 is off; the construction needs full double precision.
    def p(rho: float) -> float:
        return rho * r * t / (1.0 - b * rho) - a * rho**2

    def mu(rho: float) -> float:
        return r * t * (math.log(rho / (1.0 - b * rho)) + 1.0 / (1.0 - b * rho)) - 2.0 * a * rho

    # Spinodals: dp/drho = r t / (1 - b rho)^2 - 2 a rho = 0, either side of rho_c = 1/(3b).
    rho_c = 1.0 / (3.0 * b)
    rho_max = 1.0 / b

    def dp(rho: float) -> float:
        return r * t - 2.0 * a * rho * (1.0 - b * rho) ** 2

    tiny = _BRACKET_EPS * rho_c
    rho_sv = brentq(dp, tiny, rho_c)
    rho_sl = brentq(dp, rho_c, rho_max * (1.0 - _BRACKET_EPS))
    p_sl = p(rho_sl)

    def liquid_at(pressure: float) -> float:
        return brentq(lambda rho: p(rho) - pressure, rho_sl, rho_max * (1.0 - _BRACKET_EPS))

    # A vapour density below this has a pressure the liquid branch cannot reach.
    rho_lo = tiny if p_sl <= 0.0 else brentq(lambda rho: p(rho) - p_sl, tiny, rho_sv)
    span = rho_sv - rho_lo
    rho_v = brentq(
        lambda rho: mu(liquid_at(p(rho))) - mu(rho),
        rho_lo + _BRACKET_EPS * span,
        rho_sv - _BRACKET_EPS * span,
        xtol=1e-15,
    )
    return float(rho_v), float(liquid_at(p(rho_v)))
