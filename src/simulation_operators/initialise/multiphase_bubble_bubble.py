"""Multiphase double-bubble initialisation.

Places two low-density bubbles side-by-side in the domain surrounded
by liquid, using a smooth ``tanh`` interface profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from app_setup.registry import register_operator
from .base import InitialisationBase


@register_operator("initialise")
@dataclass
class InitialiseMultiphaseBubbleBubble(InitialisationBase):
    """Two bubbles side-by-side.

    Registered as ``"multiphase_bubble_bubble"``.
    """

    name = "multiphase_bubble_bubble"

    def __call__(
        self,
        rho_l: float = 1.0,
        rho_v: float = 0.33,
        interface_width: int = 4,
        **kwargs,
    ) -> jnp.ndarray:
        x, y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny))
        left_cx, left_cy = self.nx // 4, self.ny // 2
        right_cx, right_cy = self.nx * 2.4 // 4, self.ny // 2
        radius = min(self.nx, self.ny) // 5

        dist_left = jnp.sqrt((x - left_cx) ** 2 + (y - left_cy) ** 2)
        dist_right = jnp.sqrt((x - right_cx) ** 2 + (y - right_cy) ** 2)
        minimum_distance = jnp.minimum(dist_left, dist_right * 1.5)

        rho_field_2d = (rho_l + rho_v) / 2 + (rho_l - rho_v) / 2 * jnp.tanh(
            (minimum_distance - radius) / interface_width
        )

        rho = rho_field_2d.reshape((self.nx, self.ny, 1, 1))
        u = jnp.zeros((self.nx, self.ny, 1, 2))
        return self.equilibrium(rho, u)

