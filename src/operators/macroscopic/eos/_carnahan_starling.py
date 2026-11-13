"""Carnahan-Starling EOS: bulk chemical potential, bulk pressure and coexistence densities."""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
from src.registry import coexistence_operator
from src.registry import eos_operator
from src.registry import pressure_operator

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.config.multiphase import MultiphaseParams
    from src.protocols import EosOperator

#: Halvings per bisection. Far below float64 resolution on every bracket used
#: here, and a fixed count keeps each loop a ``fori_loop`` with a static trip count.
_N_BISECT = 200
#: Relative agreement of the two chemical potentials that counts as coexistence.
#: A solved pair agrees to ~1e-13; an end of the bracket taken for want of a root does not.
_POTENTIAL_TOLERANCE = 1e-8
#: Samples of ``dp_0/drho`` across ``(0, 4/b)`` that bracket the two spinodals.
_N_SCAN = 4096


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


def _bisect(f: Callable[[jnp.ndarray], jnp.ndarray], lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    """Root of *f* on ``[lo, hi]``, across which *f* changes sign, after ``_N_BISECT`` halvings."""

    def halve(
        _: int,
        bracket: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        low, high, f_low = bracket
        mid = 0.5 * (low + high)
        f_mid = f(mid)
        keep_low = jnp.sign(f_mid) != jnp.sign(f_low)
        return jnp.where(keep_low, low, mid), jnp.where(keep_low, mid, high), jnp.where(keep_low, f_low, f_mid)

    low, high, _ = jax.lax.fori_loop(0, _N_BISECT, halve, (lo, hi, f(lo)))
    return 0.5 * (low + high)


@jax.jit
def _maxwell_construction(
    a: float,
    b: float,
    r: float,
    t: float,
    jump: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(rho_v, rho_l, rho_s1, rho_s2)``: the coexistence pair and the two spinodals.

    Coexistence is equal chemical potential, with the liquid at the pressure
    ``p_v + jump``. ``jump = 0`` is a flat interface (the Maxwell construction);
    a curved one carries the Laplace pressure, which is the Kelvin equilibrium.
    For a trial vapour pressure ``p*`` each stable branch has one density (the
    vapour at ``p*``, the liquid at ``p* + jump``); the answer is the ``p*`` at
    which their chemical potentials agree. That residual falls strictly with
    ``p*`` (Gibbs-Duhem: its derivative is ``1/rho_l - 1/rho_v < 0``), so the
    root is unique. At ``jump = 0`` equal chemical potential is the Maxwell
    equal-area rule, since ``∫(p - p*) dv`` from ``v_l`` to ``v_v`` equals
    ``mu_l - mu_v``.

    The lower pressure bracket keeps the liquid above its spinodal pressure and
    is clamped positive: on a low isotherm the liquid spinodal pressure is
    negative, where the vapour branch has no density. Above the critical
    temperature ``dp_0/drho`` never changes sign, both spinodal indices fall to
    0, and ``rho_s1 == rho_s2`` signals it. A *jump* no pair of the two branches
    can carry leaves the chemical potentials unequal, which the caller checks.
    """

    def pressure(rho: jnp.ndarray) -> jnp.ndarray:
        return _pressure_carnahan_starling(rho, a, b, r, t)

    def potential(rho: jnp.ndarray) -> jnp.ndarray:
        return _eos_carnahan_starling(rho, a, b, r, t)

    slope = jax.grad(pressure)
    pole = jnp.asarray(4.0 / b)

    # Spinodals: the first fall of dp/drho, then the first rise after it.
    rho = jnp.linspace(1e-6 * pole, (1.0 - 1e-6) * pole, _N_SCAN)
    rising = jax.vmap(slope)(rho) > 0.0
    falls = rising[:-1] & ~rising[1:]
    i_vapour = jnp.argmax(falls)
    rises = ~rising[:-1] & rising[1:] & (jnp.arange(_N_SCAN - 1) > i_vapour)
    i_liquid = jnp.argmax(rises)
    rho_s1 = _bisect(slope, rho[i_vapour], rho[i_vapour + 1])
    rho_s2 = _bisect(slope, rho[i_liquid], rho[i_liquid + 1])

    p_hi = pressure(rho_s1)
    p_lo = jnp.maximum(pressure(rho_s2) - jump, 1e-12 * p_hi)

    def vapour(p: jnp.ndarray) -> jnp.ndarray:
        return _bisect(lambda x: pressure(x) - p, 1e-14 * rho_s1, rho_s1)

    def liquid(p: jnp.ndarray) -> jnp.ndarray:
        return _bisect(lambda x: pressure(x) - p, rho_s2, (1.0 - 1e-12) * pole)

    p_sat = _bisect(lambda p: potential(liquid(p + jump)) - potential(vapour(p)), p_lo, p_hi)
    return vapour(p_sat), liquid(p_sat + jump), rho_s1, rho_s2


@coexistence_operator(name="carnahan-starling")
def carnahan_starling_coexistence(
    *, a_eos: float, b_eos: float, r_eos: float, t_eos: float, pressure_jump: float = 0.0
) -> tuple[float, float]:
    """Coexistence densities ``(rho_v, rho_l)`` of the Carnahan-Starling EOS.

    The keywords are the EOS's row of ``EOS_PARAMETERS``, so a caller passes
    exactly that row. *pressure_jump* is ``p_l - p_v`` across the interface:
    zero for the Maxwell pair, the Laplace pressure for the Kelvin pair of a
    curved one. Precision relies on x64, which ``src`` enables at import.

    Raises:
        ValueError: If ``t_eos`` is at or above the critical temperature, where
            there is no liquid-vapour coexistence, or if no pair of the two
            branches carries *pressure_jump*.
    """
    rho_v, rho_l, rho_s1, rho_s2 = (float(x) for x in _maxwell_construction(a_eos, b_eos, r_eos, t_eos, pressure_jump))
    if not (math.isfinite(rho_s1) and math.isfinite(rho_s2) and rho_s1 < rho_s2):
        msg = (
            f"t_eos = {t_eos} is at or above the critical temperature of the Carnahan-Starling EOS "
            f"(a_eos = {a_eos}, b_eos = {b_eos}): no liquid-vapour coexistence"
        )
        raise ValueError(msg)
    mu_v, mu_l = (float(_eos_carnahan_starling(jnp.asarray(x), a_eos, b_eos, r_eos, t_eos)) for x in (rho_v, rho_l))
    if not math.isclose(mu_v, mu_l, rel_tol=_POTENTIAL_TOLERANCE):
        msg = f"the Carnahan-Starling branches have no coexisting pair with p_l - p_v = {pressure_jump}"
        raise ValueError(msg)
    return rho_v, rho_l


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
