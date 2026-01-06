from abc import ABC

import jax.numpy as jnp
from src.core.grid.grid import Grid
from src.core.lattice.lattice import Lattice


class CollisionBase(ABC):
    """
    Base class for LBM collision operators.
    Implements the BGK collision operators with source terms.
    Subclasses should implement the __call__ method.
    """

    def __init__(self, grid: Grid, lattice: Lattice) -> None:
        """
        Initializes the grid and lattice parameters required for the collision step.
        Args:
            grid (Grid): Grid object containing simulation domain information
            lattice (Lattice): Lattice object containing lattice properties
        """
        self.nx: int = grid.nx
        self.ny: int = grid.ny
        self.q: int = lattice.q
        self.d: int = lattice.d

    def __call__(self, f: jnp.ndarray, feq: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the collision step of the LBM.

        Needs to be implemented by the subclass.
        """
        pass
