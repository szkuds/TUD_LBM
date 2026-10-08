"""Shared wetting parameter container."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Mapping


class WettingParams(NamedTuple):
    """Optimisation wetting boundary parameters for hysteresis optimiser.

    Four scalar fields representing wetting behaviour at left and right contact lines.
    Used only for non-chemical-step simulations. Chemical step cases are extended with per-region pre/post variants.
    """

    d_rho_left: jnp.ndarray
    d_rho_right: jnp.ndarray
    phi_left: jnp.ndarray
    phi_right: jnp.ndarray

    @classmethod
    def from_defaults(cls, defaults: Mapping[str, float]) -> WettingParams:
        """Build the parameters from ``SimulationConfig.wetting_defaults``.

        Each of the four scalars becomes a 0-d array.
        """
        return cls(**{name: jnp.array(defaults[name]) for name in cls._fields})
