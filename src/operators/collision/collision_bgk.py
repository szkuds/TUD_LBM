from typing import TYPE_CHECKING

import jax.numpy as jnp

from .collision_base import CollisionBase

if TYPE_CHECKING:
    from config.simulation_config import SinglePhaseConfig, MultiphaseConfig


class CollisionBGK(CollisionBase):
    """
    Implements the BGK (Bhatnagar-Gross-Krook) collision operator for LBM.
    Optionally supports a source term.

    Usage:
        CollisionBGK(config=simulation_config)
    """

    def __init__(self, config: "SinglePhaseConfig | MultiphaseConfig") -> None:
        """
        Initialize the CollisionBGK operator.

        Args:
            config: Configuration object containing all simulation parameters.
        """
        super().__init__(config)
        self.tau: float = config.tau

    def __call__(
        self, f: jnp.ndarray, feq: jnp.ndarray, source: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Perform the BGK collision step.

        Args:
            f (jnp.ndarray): Distribution function.
            feq (jnp.ndarray): EquilibriumWB distribution function.
            source (jnp.ndarray, optional): Source term.

        Returns:
            jnp.ndarray: Post-collision distribution function.
        """
        if source is None:
            return (1 - (1 / self.tau)) * f + (1 / self.tau) * feq
        else:
            return (
                (1 - (1 / self.tau)) * f
                + (1 / self.tau) * feq
                + (1 - (1 / (2 * self.tau))) * source
            )
