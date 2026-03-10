"""Multiphase lateral bubble configuration initialisation.

Places two low-density bubbles stacked vertically in the domain
surrounded by liquid, using a smooth ``tanh`` interface profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from app_setup.registry import register_operator
from .base import InitialisationBase


@register_operator("initialise")
@dataclass
class InitialiseMultiphaseLateralBubble(InitialisationBase):
    """Two bubbles stacked vertically.

    Registered as ``"multiphase_lateral_bubble_configuration"``.
    """

    name = "multiphase_lateral_bubble_configuration"

    def __call__(
        self,
        rho_l: float = 1.0,
        rho_v: float = 0.33,
        interface_width: int = 4,
        **kwargs,
    ) -> jnp.ndarray:
        x, y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny))
        top_cx, top_cy = self.nx // 2, self.ny * 2 // 6
        bot_cx, bot_cy = self.nx // 2, self.ny * 4 // 6
        radius = min(self.nx, self.ny) // 6.5

        dist_top = jnp.sqrt((x - top_cx) ** 2 + (y - top_cy) ** 2)
        dist_bot = jnp.sqrt((x - bot_cx) ** 2 + (y - bot_cy) ** 2)
        minimum_distance = jnp.minimum(dist_top, dist_bot)

        rho_field_2d = (rho_l + rho_v) / 2 + (rho_l - rho_v) / 2 * jnp.tanh(
            (minimum_distance - radius) / interface_width
        )

        rho = rho_field_2d.reshape((self.nx, self.ny, 1, 1))
        u = jnp.zeros((self.nx, self.ny, 1, 2))
        return self.equilibrium(rho, u)

