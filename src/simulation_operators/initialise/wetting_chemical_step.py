"""Wetting chemical-step initialisation.

Places a droplet at the bottom wall for wetting simulations with a
chemical step, using a smooth ``tanh`` interface profile.  The droplet
is offset horizontally to one side of the domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from app_setup.registry import register_operator
from .base import InitialisationBase


@register_operator("initialise")
@dataclass
class InitialiseWettingChemicalStep(InitialisationBase):
    """Droplet wetting a solid surface with a chemical step.

    Registered as ``"wetting_chem_step"``.
    """

    name = "wetting_chem_step"

    def __call__(
        self,
        rho_l: float = 1.0,
        rho_v: float = 0.33,
        interface_width: int = 4,
        **kwargs,
    ) -> jnp.ndarray:
        r = self.ny / 3.3

        u = jnp.zeros((self.nx, self.ny, 1, 2))
        rho = jnp.zeros((self.nx, self.ny, 1, 1))

        x, y = jnp.meshgrid(
            jnp.arange(self.nx), jnp.arange(self.ny), indexing="ij"
        )

        xc = self.nx / 2

        distance = jnp.sqrt((x - xc / 2) ** 2 + y ** 2)

        rho_2d = (rho_l + rho_v) / 2 + (rho_l - rho_v) / 2 * jnp.tanh(
            2 * (r - distance) / interface_width
        )

        rho = rho.at[:, :, 0, 0].set(rho_2d)
        return self.equilibrium(rho, u)

