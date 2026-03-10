from typing import TYPE_CHECKING

import jax.numpy as jnp

from .collision_base import CollisionBase
from app_setup.registry import register_operator

if TYPE_CHECKING:
    from app_setup.simulation_config import SinglePhaseConfig, MultiphaseConfig

# Moment transformation matrix for D2Q9 lattice
M = jnp.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
    ],
)
M_INV = jnp.linalg.inv(M)


@register_operator("collision_models")
class CollisionMRT(CollisionBase):
    name = "mrt"
    """
    Implements the MRT (Multiple Relaxation Time) collision_models operator for LBM.

    Usage:
        CollisionMRT(app_setup=simulation_config)
    """

    def __init__(self, config: "SinglePhaseConfig | MultiphaseConfig") -> None:
        """
        Initialize the MRT collision_models operator.

        Args:
            config: Configuration object containing all simulation_type parameters.
        """
        super().__init__(config)
        k_diag = config.k_diag
        if k_diag is None:
            k_diag = jnp.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.8, 0.8])
        self.K = k_diag

    def __call__(
        self, f: jnp.ndarray, feq: jnp.ndarray, source: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Perform the MRT collision_models step.

        Args:
            f (jnp.ndarray): Distribution function.
            feq (jnp.ndarray): EquilibriumWB distribution function.
            source (jnp.ndarray, optional): Source term.

        Returns:
            jnp.ndarray: Post-collision_models distribution function.
        """
        K = jnp.diag(self.K)
        I = jnp.eye(len(K))
        # Transform to moment space
        mat_f_neq = M_INV @ K @ M
        mat_source = M_INV @ (I - K / 2) @ M
        f_neq_post = jnp.einsum("ij,xyj->xyi", mat_f_neq, (feq - f)[..., 0])
        source_post = jnp.einsum("ij,xyj->xyi", mat_source, source[..., 0])
        f_post = f[..., 0] + f_neq_post + source_post
        return f_post[..., None]
