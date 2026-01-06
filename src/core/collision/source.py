import jax.numpy as jnp
from src.core.grid.grid import Grid
from src.core.lattice.lattice import Lattice
from src.operators.differential import Gradient


class SourceTerm:
    """
    Callable class to calculate the source term for the LBM equation.
    """

    def __init__(self, grid: Grid, lattice: Lattice, bc_config: dict = None):
        """
        Initialize the source term calculator.

        Args:
            grid (Grid): Grid object containing simulation domain information
            lattice (Lattice): Lattice object containing lattice properties
            bc_config (dict, optional): Boundary condition configuration
        """
        self.nx: int = grid.nx
        self.ny: int = grid.ny
        self.q: int = lattice.q
        self.d: int = lattice.d
        self.w = lattice.w
        self.cx = lattice.c[0]
        self.cy = lattice.c[1]
        self.bc_config = bc_config
        self.gradient = Gradient(lattice, bc_config=bc_config)

    def __call__(
        self, rho: jnp.ndarray, u: jnp.ndarray, force: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the source term for the LBM equation.

        Args:
            rho (jnp.ndarray): Density field, shape (nx, ny, 1, 1)
            u (jnp.ndarray): Velocity field, shape (nx, ny, 1, 2)
            force (jnp.ndarray): Force field, shape (nx, ny, 1, 2)

        Returns:
            jnp.ndarray: Source term, shape (nx, ny, q, 1)
        """
        nx, ny, q = self.nx, self.ny, self.q
        w = self.w
        c = (self.cx, self.cy)
        d = self.d
        gradient = self.gradient

        # Extract 2D data from 4D arrays
        u_2d = u[:, :, 0, :]  # Shape: (nx, ny, 2)
        force_2d = force[:, :, 0, :]  # Shape: (nx, ny, 2)

        # Calculate gradient of rho
        grad_rho = self.gradient(rho)
        grad_rho_2d = grad_rho[:, :, 0, :]  # Shape: (nx, ny, 2)

        def source_term(u_2d_, force_2d_, grad_rho_2d_):
            """
            Compute the source term for each lattice direction.

            Args:
                u_2d_ (jnp.ndarray): Velocity field, shape (nx, ny, 2)
                force_2d_ (jnp.ndarray): Force field, shape (nx, ny, 2)
                grad_rho_2d_ (jnp.ndarray): Gradient of density, shape (nx, ny, 2)

            Returns:
                jnp.ndarray: Source term, shape (nx, ny, q)
            """
            cx, cy = c[0], c[1]
            fx, fy = force_2d_[:, :, 0], force_2d_[:, :, 1]
            ux, uy = u_2d_[:, :, 0], u_2d_[:, :, 1]
            grad_rho_x, grad_rho_y = grad_rho_2d_[:, :, 0], grad_rho_2d_[:, :, 1]

            fx_cor = fx + (grad_rho_x / 3)
            fy_cor = fy + (grad_rho_y / 3)
            source_ = jnp.zeros((nx, ny, q))

            for i in range(q):
                # Compute the source term for each direction
                source_ = source_.at[:, :, i].set(
                    w[i]
                    * (
                        3 * (cx[i] * fx + cy[i] * fy)
                        + 9
                        * (cx[i] * fx_cor + cy[i] * fy_cor)
                        * (cx[i] * ux + cy[i] * uy)
                        - 3 * (ux * fx_cor + uy * fy_cor)
                        + 0.5
                        * (3 * (cx[i] * cx[i] + cy[i] * cy[i]) - d)
                        * (ux * grad_rho_x + uy * grad_rho_y)
                    )
                )

            return source_

        source_3d = source_term(u_2d, force_2d, grad_rho_2d)

        # Convert to 4D format: (nx, ny, q, 1)
        source_4d = jnp.expand_dims(source_3d, axis=-1)

        return source_4d
