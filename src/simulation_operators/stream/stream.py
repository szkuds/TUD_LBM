#   TODO: Will need to be adapted for 3D
from typing import TYPE_CHECKING

import jax.numpy as jnp

from app_setup.registry import register_operator

if TYPE_CHECKING:
    from app_setup.simulation_setup import SimulationSetup


@register_operator("stream")
class Streaming:
    name = "standard"
    """
    Callable class to perform the streaming step of the LBM.

    Usage:
        Streaming(app_setup=simulation_config)
    """

    def __init__(self, config: "SimulationSetup") -> None:
        """
        Initialize the Streaming operator.

        Args:
            config: Configuration object containing all simulation_type parameters.
        """
        from simulation_domain.lattice import Lattice

        lattice = Lattice(config.lattice_type)
        self.c = lattice.c  # Shape: (2, Q)
        self.q = lattice.q

    def __call__(self, f):
        """
        Perform the streaming step of the LBM.

        Args:
            f (jnp.ndarray): Distribution function, shape (nx, ny, q, 1)

        Returns:
            jnp.ndarray: Post-streaming distribution function.
        """
        for i in range(self.q):
            f = f.at[:, :, i, 0].set(
                jnp.roll(
                    jnp.roll(f[:, :, i, 0], self.c[0, i], axis=0), self.c[1, i], axis=1
                )
            )
        return f
