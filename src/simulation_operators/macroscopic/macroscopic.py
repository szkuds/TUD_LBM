from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import jit, Array

from app_setup.registry import register_operator

if TYPE_CHECKING:
    from app_setup.simulation_setup import SimulationSetup


@register_operator("macroscopic")
class Macroscopic:
    name = "standard"
    """
    Calculates the macroscopic density and velocity fields from the population distribution.

    Usage:
        Macroscopic(app_setup=simulation_config)
    """

    def __init__(self, config: "SimulationSetup") -> None:
        """
        Initialize the Macroscopic operator.

        Args:
            config: Configuration object containing all simulation_type parameters.
        """
        from simulation_domain.grid import Grid
        from simulation_domain.lattice import Lattice

        grid = Grid(config.grid_shape)
        lattice = Lattice(config.lattice_type)

        self.nx: int = grid.nx
        self.ny: int = grid.ny
        self.q: int = lattice.q
        self.d: int = lattice.d
        self.cx: jnp.ndarray = jnp.array(lattice.c[0])
        self.cy: jnp.ndarray = jnp.array(lattice.c[1])
        self.force_enabled = config.force_enabled

    @partial(jit, static_argnums=(0,))
    def __call__(
        self, f: jnp.ndarray, force: jnp.ndarray = None
    ) -> tuple[Array, Array, Array] | tuple[Array, Array] | None:
        """
        Args:
            f (jnp.ndarray): Population distribution, shape (nx, ny, q, 1)
            force (jnp.ndarray, optional): Force field, shape (nx, ny, 1, 2). Required if force_enabled is True.

        Returns:
            tuple: (rho, u)
                rho (jnp.ndarray): Density field, shape (nx, ny, 1, 1)
                u (jnp.ndarray): Velocity field, shape (nx, ny, 1, 2)
        """
        if self.d == 2:
            # Compute density
            rho = jnp.sum(f, axis=2, keepdims=True)  # (nx, ny, 1, 1)
            # Compute velocity WITHOUT force correction
            cx = self.cx.reshape((1, 1, self.q, 1))
            cy = self.cy.reshape((1, 1, self.q, 1))
            ux = jnp.sum(f * cx, axis=2, keepdims=True)
            uy = jnp.sum(f * cy, axis=2, keepdims=True)
            u = jnp.concatenate([ux, uy], axis=-1) / rho  # (nx, ny, 1, 2)

            if force is not None:
                u_eq = u + force / (2 * rho)
                return rho, u_eq, force
            if force is None:
                return rho, u
        elif self.d == 3:
            raise NotImplementedError("Dimension larger than 2 not supported.")
