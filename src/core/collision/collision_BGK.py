import jax.numpy as jnp
from src.core.grid.grid import Grid
from src.core.lattice.lattice import Lattice
from .base import CollisionBase


class CollisionBGK(CollisionBase):
    """
    Implements the BGK (Bhatnagar-Gross-Krook) collision operator for LBM.
    Optionally supports a source term.
    """

    def __init__(self, grid: Grid, lattice: Lattice, tau: float) -> None:
        """
        Initialize the CollisionBGK operator.

        Args:
            grid (Grid): Grid object containing simulation domain information
            lattice (Lattice): Lattice object containing lattice properties
            tau (float): Relaxation time parameter
        """
        super().__init__(grid, lattice)
        self.tau: float = tau

    def __call__(
        self, f: jnp.ndarray, feq: jnp.ndarray, source: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Perform the BGK collision step.

        Args:
            f (jnp.ndarray): Distribution function.
            feq (jnp.ndarray): Equilibrium distribution function.
            source (jnp.ndarray, optional): Source term.

        Returns:
            jnp.ndarray: Post-collision distribution function.
        """
        if source is None:
            # Standard BGK collision without source term
            return (1 - (1 / self.tau)) * f + (1 / self.tau) * feq
        else:
            # BGK collision with source term
            return (
                (1 - (1 / self.tau)) * f
                + (1 / self.tau) * feq
                + (1 - (1 / (2 * self.tau))) * source
            )
