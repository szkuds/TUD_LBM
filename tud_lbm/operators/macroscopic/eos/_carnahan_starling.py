"""Carnahan-Starling EOS callable builder."""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from tud_lbm.registry import eos_operator

if TYPE_CHECKING:
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


@eos_operator(name="carnahan-starling")
def build_carnahan_starling_eos(mp: MultiphaseParams) -> EOSFunction:
    """Return ``eos_fn(rho)`` for the Carnahan-Starling EOS using bound params."""
    return lambda rho: _eos_carnahan_starling(rho, mp.a_eos, mp.b_eos, mp.r_eos, mp.t_eos)
