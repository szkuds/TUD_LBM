"""Shared wetting parameter containers."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple

if TYPE_CHECKING:
    import jax.numpy as jnp


class WettingParams(NamedTuple):
    """Optimisation wetting boundary parameters for hysteresis optimiser.

    Four scalar fields representing wetting behaviour at left and right contact lines.
    Used only for non-chemical-step simulations. Chemical step cases are extended with per-region pre/post variants.
    """

    d_rho_left: jnp.ndarray
    d_rho_right: jnp.ndarray
    phi_left: jnp.ndarray
    phi_right: jnp.ndarray
