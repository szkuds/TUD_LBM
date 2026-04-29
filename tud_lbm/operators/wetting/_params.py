"""Shared wetting parameter containers."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple

if TYPE_CHECKING:
    import jax.numpy as jnp


class WettingParams(NamedTuple):
    """Optimisable wetting boundary parameters used across wetting operators."""

    # Per-region (pre/post) parameters for left/bottom contact line
    d_rho_left_pre: jnp.ndarray
    d_rho_left_post: jnp.ndarray
    phi_left_pre: jnp.ndarray
    phi_left_post: jnp.ndarray

    # Per-region (pre/post) parameters for right/top contact line
    d_rho_right_pre: jnp.ndarray
    d_rho_right_post: jnp.ndarray
    phi_right_pre: jnp.ndarray
    phi_right_post: jnp.ndarray
