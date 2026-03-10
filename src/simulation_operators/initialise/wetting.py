"""Wetting initialisation.

Places a droplet at the bottom wall for wetting simulations, using a
smooth ``tanh`` interface profile centred horizontally and anchored at
the lower boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from app_setup.registry import register_operator
from .base import InitialisationBase


@register_operator("initialise")
@dataclass
class InitialiseWetting(InitialisationBase):
    """Droplet wetting a solid surface (bottom wall).

    Registered as ``"wetting"``.
    """

    name = "wetting"

    def __call__(
        self,
        rho_l: float = 1.0,
        rho_v: float = 0.33,
        interface_width: int = 4,
        **kwargs,
    ) -> jnp.ndarray:
        r = self.ny / 3.33

        u = jnp.zeros((self.nx, self.ny, 1, 2))
        rho = jnp.zeros((self.nx, self.ny, 1, 1))

        x, y = jnp.meshgrid(
            jnp.arange(self.nx), jnp.arange(self.ny), indexing="ij"
        )

        xc = self.nx / 2

        distance = jnp.sqrt((x - xc) ** 2 + y ** 2)

        rho_2d = (rho_l + rho_v) / 2 + (rho_l - rho_v) / 2 * jnp.tanh(
            2 * (r - distance) / interface_width
        )

        rho = rho.at[:, :, 0, 0].set(rho_2d)
        return self.equilibrium(rho, u)

