"""Carnahan-Starling EOS callable builder."""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from tud_lbm.registry import eos_operator

if TYPE_CHECKING:
    import numpy as np
    from tud_lbm.operators.macroscopic import MultiphaseParams
    from tud_lbm.operators.protocols import EOSFunction


def _eos_carnahan_starling(
    rho: jnp.ndarray,
    a: float,
    b: float,
    r: float,
    t: float,
) -> jnp.ndarray:
    """Carnahan-Starling EOS bulk chemical potential ``mu_0(rho)``."""
    return -2.0 * a * rho + r * t * (1.0 + jnp.log(rho)) + (16.0 * r * t * (b * rho - 12.0)) / (b * rho - 4.0) ** 3


def carnahan_starling_pressure(
    rho: jnp.ndarray | np.ndarray,
    a: float,
    b: float,
    r: float,
    t: float,
) -> jnp.ndarray | np.ndarray:
    """Carnahan-Starling bulk thermodynamic pressure ``p_0(rho)``.

    Consistent with ``_eos_carnahan_starling`` (the two differ only by an
    additive constant, which cancels in the Laplace pressure jump). Used by
    the surface-tension calibration; not part of the force pipeline. Accepts
    NumPy or JAX arrays.
    """
    eta = b * rho / 4.0
    ideal = rho * r * t * (1.0 + eta + eta**2 - eta**3) / (1.0 - eta) ** 3
    return ideal - a * rho**2


@eos_operator(name="carnahan-starling")
def build_carnahan_starling_eos(mp: MultiphaseParams) -> EOSFunction:
    """Return ``eos_fn(rho)`` for the Carnahan-Starling EOS using bound params."""
    if mp.a_eos is None or mp.b_eos is None or mp.r_eos is None or mp.t_eos is None:
        msg = "a_eos, b_eos, r_eos, t_eos are all required for Carnahan-Starling EOS"
        raise ValueError(msg)
    a, b, r, t = mp.a_eos, mp.b_eos, mp.r_eos, mp.t_eos
    return lambda rho: _eos_carnahan_starling(rho, a, b, r, t)
