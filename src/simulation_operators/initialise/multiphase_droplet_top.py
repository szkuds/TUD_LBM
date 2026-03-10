"""Multiphase droplet-at-top initialisation.

Places a high-density droplet near the top of the domain surrounded
by vapour, using a smooth ``tanh`` interface profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from app_setup.registry import register_operator
from .base import InitialisationBase


@register_operator("initialise")
@dataclass
class InitialiseMultiphaseDropletTop(InitialisationBase):
    """Droplet near the top of the domain.

    Registered as ``"multiphase_droplet_top"``.
    """

    name = "multiphase_droplet_top"

    def __call__(
        self,
        rho_l: float = 1.0,
        rho_v: float = 0.33,
        interface_width: int = 4,
        **kwargs,
    ) -> jnp.ndarray:
        x, y = jnp.meshgrid(
            jnp.arange(self.nx), jnp.arange(self.ny), indexing="ij"
        )
        center_x, center_y = self.nx // 2, 5 * self.ny // 6
        radius = min(self.nx, self.ny) // 4

        distance = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        rho_field_2d = (rho_l + rho_v) / 2 - (rho_l - rho_v) / 2 * jnp.tanh(
            (distance - radius) / interface_width
        )

        rho = rho_field_2d.reshape((self.nx, self.ny, 1, 1))
        u = jnp.zeros((self.nx, self.ny, 1, 2))
        return self.equilibrium(rho, u)

