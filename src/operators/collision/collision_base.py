from abc import ABC, abstractmethod

import jax.numpy as jnp
from domain.grid.grid import Grid
from domain.lattice.lattice import Lattice


class CollisionBase(ABC):
    """
    Base class for LBM collision operators.
    Implements the BGK collision operators with source terms.

    Subclasses should implement the __call__ method following the
    CollisionOperator protocol (see operators.protocols).
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

    @abstractmethod
    def __call__(
        self, f: jnp.ndarray, feq: jnp.ndarray, source: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """
        Perform the collision step of the LBM.

        Args:
            f: Distribution function, shape (nx, ny, q, 1)
            feq: Equilibrium distribution function, shape (nx, ny, q, 1)
            source: Optional source term, shape (nx, ny, q, 1)

        Returns:
            Post-collision distribution function, shape (nx, ny, q, 1)
        """
        ...
