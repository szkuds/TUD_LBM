"""Shared wetting parameter containers."""

from __future__ import annotations
from typing import NamedTuple
import jax.numpy as jnp


class WettingParams(NamedTuple):
    """Optimisable wetting boundary parameters used across wetting operators."""

    d_rho_left: jnp.ndarray
    d_rho_right: jnp.ndarray
    phi_left: jnp.ndarray
    phi_right: jnp.ndarray
