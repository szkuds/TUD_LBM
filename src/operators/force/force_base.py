from abc import ABC, abstractmethod
import jax.numpy as jnp


class Force(ABC):
    """
    Abstract base class for forces in the simulation.
    Holds a force array of shape (nx, ny, 1, d).
    """

    def __init__(self, force_array: jnp.ndarray):
        if force_array.ndim != 4 or force_array.shape[2] != 1:
            raise ValueError("Force array must have shape (nx, ny, 1, d)")
        self.force = force_array

    @abstractmethod
    @abstractmethod
    def compute_force(self, **kwargs) -> jnp.ndarray:
        """
        Compute the force field.

        Each force extracts only required parameters from kwargs.
        Must return array of shape (nx, ny, 1, d).

        Common kwargs:
            rho: Density field (nx, ny, 1, 1)
            rho_l, rho_v: Phase densities (float)
            hi: Electric potential distribution (nx, ny, q, 1)
        """
        pass
