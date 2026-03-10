"""Standard single-phase initialisation.

Initialises a uniform density and velocity field and returns
the corresponding equilibrium distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from app_setup.registry import register_operator
from .base import InitialisationBase


@register_operator("initialise")
@dataclass
class StandardInitialisation(InitialisationBase):
    """Standard single-phase initialisation (uniform density and velocity).

    Registered as ``"standard"`` — the default for single-phase simulations.
    """

    name = "standard"

    def __call__(
        self,
        density: float = 1.0,
        velocity: np.ndarray = np.array([0.0, 0.0]),
        **kwargs,
    ) -> jnp.ndarray:
        """Return equilibrium distribution for uniform *density* and *velocity*.

        Args:
            density: Initial uniform density.
            velocity: Initial uniform velocity ``[ux, uy]``.

        Returns:
            4-D JAX array ``f``.
        """
        rho = jnp.full((self.nx, self.ny, 1, 1), density)
        u = jnp.broadcast_to(
            jnp.array(velocity).reshape(1, 1, 1, 2), (self.nx, self.ny, 1, 2)
        )
        return self.equilibrium(rho, u)

