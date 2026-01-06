import jax.numpy as jnp
from .base import CollisionBase
from src.core.grid.grid import Grid
from src.core.lattice.lattice import Lattice

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


class CollisionMRT(CollisionBase):
    """
    Implements the MRT (Multiple Relaxation Time) collision operator for LBM.
    """

    def __init__(self, grid: Grid, lattice: Lattice, k_diag=None, **kwargs):
        """
        Initialize the MRT collision operator.

        Args:
            grid (Grid): Grid object containing simulation domain information
            lattice (Lattice): Lattice object containing lattice properties
            k_diag (jnp.ndarray, optional): Diagonal relaxation rates for moments.
            kwargs: Optional relaxation rates for specific moments.
        """
        super().__init__(grid, lattice)
        if k_diag is None:
            k_diag = jnp.array(
                [
                    kwargs.get("k0", 0.0),
                    kwargs.get("kb", 1.0),
                    kwargs.get("k2", 1.0),
                    kwargs.get("k0", 0.0),
                    kwargs.get("k4", 1.0),
                    kwargs.get("k0", 0.0),
                    kwargs.get("k4", 1.0),
                    kwargs.get("kv", 0.8),
                    kwargs.get("kv", 0.8),
                ]
            )
        self.K = k_diag

    def __call__(
        self, f: jnp.ndarray, feq: jnp.ndarray, source: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Perform the MRT collision step.

        Args:
            f (jnp.ndarray): Distribution function.
            feq (jnp.ndarray): Equilibrium distribution function.
            source (jnp.ndarray, optional): Source term.

        Returns:
            jnp.ndarray: Post-collision distribution function.
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
