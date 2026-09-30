"""EOS parameters for a target density ratio, liquid density, liquid sound speed and interface width.

The study compares equations of state at one density ratio, so each EOS must
be placed on the same footing. Four targets fix everything:

* ``ratio`` = ``rho_l / rho_v`` on the coexistence curve (Maxwell construction),
  which sets the reduced temperature ``T_r``;
* ``rho_l``, which sets the density scale ``b`` (at fixed ``T_r`` every
  coexistence density scales as ``1/b``);
* ``c_l`` = ``sqrt(dp_0/drho)`` at ``rho_l``, which sets the pressure scale ``a``
  (at fixed ``T_r`` and ``b``, ``p_0`` is linear in ``a``);
* ``width``, the planar-interface width ``drho / max|drho/dx|``, which sets ``kappa``.

The double-well is its own reference: ``interface_width`` is exactly this width,
and its ``c_l**2 = 16 kappa / W**2`` follows from ``kappa`` and ``W``. With ``c_l``
matched across EOSs, the vapour sound speed ``c_v`` is the variable the EOS axis
isolates.

The planar interface also gives the surface tension ``sigma = int sqrt(2 kappa
omega) drho``, with ``omega(rho) = int_{rho_v}^{rho} (mu_0 - mu_sat)`` the excess
grand-potential density. For the double-well it reproduces the closed form
``(2/3)(kappa/W) drho**2``.
"""

from __future__ import annotations
import math
from collections.abc import Callable
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from src.operators.macroscopic.eos._carnahan_starling import _cs_dp_drho
from src.operators.macroscopic.eos._carnahan_starling import _eos_carnahan_starling
from src.operators.macroscopic.eos._carnahan_starling import coexistence_densities as cs_coexistence
from src.operators.macroscopic.eos._double_well import _eos_double_well
from src.operators.macroscopic.eos._double_well import _pressure_double_well
from src.operators.macroscopic.eos._van_der_waals import _eos_van_der_waals
from src.operators.macroscopic.eos._van_der_waals import coexistence_densities as vdw_coexistence

jax.config.update("jax_enable_x64", True)

#: Critical temperature over ``a/(b R)``: ``8/27`` for vdW, ``0.3773`` for CS.
_TC_FACTOR = {"van-der-waals": 8.0 / 27.0, "carnahan-starling": 0.3773}
_COEXISTENCE = {"van-der-waals": vdw_coexistence, "carnahan-starling": cs_coexistence}
_MU = {"van-der-waals": _eos_van_der_waals, "carnahan-starling": _eos_carnahan_starling}

MuFn = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class EosPoint:
    """One EOS at its coexistence point, with the derived study quantities."""

    eos: str
    config: dict[str, float | str]
    rho_v: float
    rho_l: float
    c_v: float
    c_l: float
    width: float
    sigma: float
    t_r: float | None = None

    def row(self) -> dict[str, float | str | None]:
        """The descriptive quantities, as one results-table row."""
        return {
            "eos": self.eos,
            "rho_v": self.rho_v,
            "rho_l": self.rho_l,
            "ratio": self.rho_l / self.rho_v,
            "c_v": self.c_v,
            "c_l": self.c_l,
            "width": self.width,
            "sigma": self.sigma,
            "T_r": self.t_r,
        }


def _dp_drho(eos: str, rho: float, a: float, b: float, r: float, t: float) -> float:
    if eos == "carnahan-starling":
        return float(_cs_dp_drho(rho, a, b, r, t))
    return r * t / (1.0 - b * rho) ** 2 - 2.0 * a * rho


def _omega(mu: MuFn, rho_v: float, rho_l: float, n: int = 20_001) -> tuple[np.ndarray, np.ndarray]:
    """Excess grand-potential density ``omega(rho)`` on ``[rho_v, rho_l]``."""
    # Geometric spacing near rho_v: the vapour side of a high-ratio profile spans decades.
    rho = np.unique(np.concatenate([np.geomspace(rho_v, rho_l, n), np.linspace(rho_v, rho_l, n)]))
    mu_vals = np.asarray(mu(jnp.asarray(rho)), dtype=np.float64)
    mu_sat = mu_vals[0]
    omega = cumulative_trapezoid(mu_vals - mu_sat, rho, initial=0.0)
    return rho, np.maximum(omega, 0.0)


def planar_interface(mu: MuFn, rho_v: float, rho_l: float, kappa: float) -> tuple[float, float]:
    """``(width, sigma)`` of the planar interface of ``mu_0`` with gradient coefficient ``kappa``."""
    rho, omega = _omega(mu, rho_v, rho_l)
    width = (rho_l - rho_v) / math.sqrt(2.0 * float(omega.max()) / kappa)
    sigma = float(np.trapezoid(np.sqrt(2.0 * kappa * omega), rho))
    return width, sigma


def _kappa_for_width(mu: MuFn, rho_v: float, rho_l: float, width: float) -> float:
    """``kappa`` giving planar width *width*: ``width**2 * 2 omega_max / drho**2``."""
    _, omega = _omega(mu, rho_v, rho_l)
    return width**2 * 2.0 * float(omega.max()) / (rho_l - rho_v) ** 2


def double_well(rho_l: float, rho_v: float, kappa: float, width: int) -> EosPoint:
    """The double-well at its configured parameters."""
    beta = 8.0 * kappa / (width**2 * (rho_l - rho_v) ** 2)

    def mu(rho: jnp.ndarray) -> jnp.ndarray:
        return _eos_double_well(rho, beta, rho_l, rho_v)

    def p(rho: jnp.ndarray) -> jnp.ndarray:
        return _pressure_double_well(rho, beta, rho_l, rho_v)

    dp = jax.grad(p)
    w, sigma = planar_interface(mu, rho_v, rho_l, kappa)
    return EosPoint(
        eos="double-well",
        config={"eos": "double-well", "kappa": kappa, "rho_l": rho_l, "rho_v": rho_v, "interface_width": width},
        rho_v=rho_v,
        rho_l=rho_l,
        c_v=math.sqrt(float(dp(rho_v))),
        c_l=math.sqrt(float(dp(rho_l))),
        width=w,
        sigma=sigma,
    )


def cubic_eos(
    eos: str,
    ratio: float,
    rho_l: float | None,
    c_l: float | None,
    width: float,
    *,
    a: float | None = None,
    b: float | None = None,
) -> EosPoint:
    """Van der Waals or Carnahan-Starling at *ratio* and planar *width* (module docstring).

    The density scale is *b* when given, else derived from *rho_l*; the pressure scale is
    *a* when given, else derived from *c_l*. Fixing both reproduces a literature set
    (e.g. CS ``a = 1, b = 4``) at the temperature that gives *ratio*.
    """
    coexist = _COEXISTENCE[eos]
    r = 1.0

    def t_of(a: float, b: float, t_r: float) -> float:
        return t_r * _TC_FACTOR[eos] * a / (b * r)

    # 1. Reduced temperature from the ratio (a = b = 1; the ratio depends on T_r only).
    def log_ratio_mismatch(t_r: float) -> float:
        rv, rl = coexist(1.0, 1.0, r, t_of(1.0, 1.0, t_r))
        return math.log(rl / rv) - math.log(ratio)

    t_r = brentq(log_ratio_mismatch, 0.2, 0.99, xtol=1e-12)
    # 2. Density scale: coexistence densities scale as 1/b.
    _, rl_unit = coexist(1.0, 1.0, r, t_of(1.0, 1.0, t_r))
    if b is None:
        assert rho_l is not None  # noqa: S101 - one of b / rho_l is required
        b = rl_unit / rho_l
    # 3. Pressure scale: dp/drho at rho_l is linear in a at fixed T_r and b.
    if a is None:
        assert c_l is not None  # noqa: S101 - one of a / c_l is required
        _, rl1 = coexist(1.0, b, r, t_of(1.0, b, t_r))
        a = c_l**2 / _dp_drho(eos, rl1, 1.0, b, r, t_of(1.0, b, t_r))
    t = t_of(a, b, t_r)
    rho_v_eq, rho_l_eq = coexist(a, b, r, t)

    def mu(rho: jnp.ndarray) -> jnp.ndarray:
        return _MU[eos](rho, a, b, r, t)

    # 4. Gradient coefficient from the width.
    kappa = _kappa_for_width(mu, rho_v_eq, rho_l_eq, width)
    w, sigma = planar_interface(mu, rho_v_eq, rho_l_eq, kappa)
    return EosPoint(
        eos=eos,
        config={
            "eos": eos,
            "kappa": kappa,
            "rho_l": rho_l_eq,
            "rho_v": rho_v_eq,
            # Initialisation hint only for a cubic EOS. The same value as the
            # double-well's W, so every EOS starts from the same tanh profile.
            "interface_width": round(width),
            "a_eos": a,
            "b_eos": b,
            "r_eos": r,
            "t_eos": t,
        },
        rho_v=rho_v_eq,
        rho_l=rho_l_eq,
        c_v=math.sqrt(_dp_drho(eos, rho_v_eq, a, b, r, t)),
        c_l=math.sqrt(_dp_drho(eos, rho_l_eq, a, b, r, t)),
        width=w,
        sigma=sigma,
        t_r=t_r,
    )


if __name__ == "__main__":
    for point in (
        double_well(1.0, 1e-3, 0.04, 5),
        double_well(1.0, 1e-3, 0.16, 5),
        cubic_eos("van-der-waals", 1000.0, 1.0, 0.16, 5.0),
        cubic_eos("carnahan-starling", 1000.0, 1.0, 0.16, 5.0),
        cubic_eos("carnahan-starling", 1000.0, None, None, 5.0, a=1.0, b=4.0),
        cubic_eos("van-der-waals", 1000.0, None, None, 5.0, a=9.0 / 392.0, b=2.0 / 21.0),
        cubic_eos("carnahan-starling", 1000.0, None, 0.3, 5.0, b=4.0),
        cubic_eos("van-der-waals", 1000.0, None, 0.3, 5.0, b=2.0 / 21.0),
    ):
        print(point.row(), point.config)
