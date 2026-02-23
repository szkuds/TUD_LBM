from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from config.simulation_config import SinglePhaseConfig, MultiphaseConfig


class CollisionBase(ABC):
    """
    Base class for LBM collision operators.
    Implements the BGK collision operators with source terms.

    Subclasses should implement the __call__ method following the
    CollisionOperator protocol (see operators.protocols).

    Usage:
        CollisionBGK(config=simulation_config)
    """

    def __init__(self, config: "SinglePhaseConfig | MultiphaseConfig") -> None:
        """
        Initializes the grid and lattice parameters required for the collision step.

        Args:
            config: Configuration object containing all simulation parameters.
        """
        from domain.grid import Grid
        from domain.lattice import Lattice

        grid = Grid(config.grid_shape)
        lattice = Lattice(config.lattice_type)
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
