"""Double-well EOS callable builder."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tud_lbm.registry import eos_operator

if TYPE_CHECKING:
    import jax.numpy as jnp
    from tud_lbm.operators.macroscopic import MultiphaseParams
    from tud_lbm.operators.protocols import EOSFunction


def _eos_double_well(
    rho: jnp.ndarray,
    beta: float,
    rho_l: float,
    rho_v: float,
) -> jnp.ndarray:
    """Double-well equation-of-state derivative for bulk chemical potential."""
    return 2.0 * beta * (rho - rho_l) * (rho - rho_v) * (2.0 * rho - rho_l - rho_v)


@eos_operator(name="double-well")
def build_double_well_eos(mp: MultiphaseParams) -> EOSFunction:
    """Return ``eos_fn(rho)`` for the double-well EOS using bound params."""
    beta = 8.0 * mp.kappa / (float(mp.interface_width) ** 2 * (mp.rho_l - mp.rho_v) ** 2)
    return lambda rho: _eos_double_well(rho, beta, mp.rho_l, mp.rho_v)
