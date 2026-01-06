from functools import partial

import jax.numpy as jnp
from jax import jit, Array
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.grid.grid import Grid
    from src.core.lattice.lattice import Lattice


class Macroscopic:
    """
    Calculates the macroscopic density and velocity fields from the population distribution.
    """

    def __init__(
        self, grid: "Grid", lattice: "Lattice", force_enabled: bool = False
    ) -> None:
        self.nx: int = grid.nx
        self.ny: int = grid.ny
        self.q: int = lattice.q
        self.d: int = lattice.d
        self.cx: jnp.ndarray = jnp.array(lattice.c[0])
        self.cy: jnp.ndarray = jnp.array(lattice.c[1])
        self.force_enabled = force_enabled

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
