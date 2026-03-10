from typing import TYPE_CHECKING

import jax.numpy as jnp

from .collision_base import CollisionBase
from app_setup.registry import register_operator

if TYPE_CHECKING:
    from app_setup.simulation_setup import SimulationSetup


@register_operator("collision_models")
class CollisionBGK(CollisionBase):
    name = "bgk"
    """
    Implements the BGK (Bhatnagar-Gross-Krook) collision_models operator for LBM.
    Optionally supports a source term.

    Usage:
        CollisionBGK(app_setup=simulation_config)
    """

    def __init__(self, config: "SimulationSetup") -> None:
        """
        Initialize the CollisionBGK operator.

        Args:
            config: Configuration object containing all simulation_type parameters.
        """
        super().__init__(config)
        self.tau: float = config.tau

    def __call__(
        self, f: jnp.ndarray, feq: jnp.ndarray, source: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Perform the BGK collision_models step.

        Args:
            f (jnp.ndarray): Distribution function.
            feq (jnp.ndarray): EquilibriumWB distribution function.
            source (jnp.ndarray, optional): Source term.

        Returns:
            jnp.ndarray: Post-collision_models distribution function.
        """
        if source is None:
            return (1 - (1 / self.tau)) * f + (1 / self.tau) * feq
        else:
            return (
                (1 - (1 / self.tau)) * f
                + (1 / self.tau) * feq
                + (1 - (1 / (2 * self.tau))) * source
            )
