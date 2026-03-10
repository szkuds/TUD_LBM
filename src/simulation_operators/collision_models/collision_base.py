from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from app_setup.simulation_setup import SimulationSetup


class CollisionBase(ABC):
    """
    Base class for LBM collision_models simulation_operators.
    Implements the BGK collision_models simulation_operators with source terms.

    Subclasses should implement the __call__ method following the
    CollisionOperator protocol (see simulation_operators.protocols).

    Usage:
        CollisionBGK(app_setup=simulation_config)
    """

    def __init__(self, config: "SimulationSetup") -> None:
        """
        Initializes the grid and lattice parameters required for the collision_models step.

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

    @abstractmethod
    def __call__(
        self, f: jnp.ndarray, feq: jnp.ndarray, source: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """
        Perform the collision_models step of the LBM.

        Args:
            f: Distribution function, shape (nx, ny, q, 1)
            feq: Equilibrium distribution function, shape (nx, ny, q, 1)
            source: Optional source term, shape (nx, ny, q, 1)

        Returns:
            Post-collision_models distribution function, shape (nx, ny, q, 1)
        """
        ...
