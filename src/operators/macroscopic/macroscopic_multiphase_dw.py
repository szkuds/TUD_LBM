from functools import partial

import jax.numpy as jnp
from jax import jit

from src.core.lattice.lattice import Lattice
from src.core.grid.grid import Grid
from src.operators.differential import Gradient, Laplacian
from src.operators.wetting import determine_padding_modes
from .macroscopic import Macroscopic



class MacroscopicMultiphaseDW(Macroscopic):
    """
    Calculates macroscopic variables for multiphase simulations.
    Inherits from Macroscopic and adds multiphase-specific methods.
    This is the double well implementation.
    """

    def __init__(
        self,
        grid: Grid,
        lattice: Lattice,
        kappa: float,
        interface_width: int,
        rho_l: float,
        rho_v: float,
        force_enabled: bool = False,
        bc_config: dict = None,
    ):
        super().__init__(
            grid, lattice, force_enabled=force_enabled
        )
        self.kappa = kappa
        self.rho_l = rho_l
        self.rho_v = rho_v
        self.bc_config = bc_config
        self.gradient = Gradient(lattice, bc_config=bc_config)
        self.laplacian = Laplacian(lattice, bc_config=bc_config)
        self.beta = 8 * kappa / (float(interface_width) ** 2 * (rho_l - rho_v) ** 2)

    @partial(jit, static_argnums=(0,))
    def __call__(
        self, f: jnp.ndarray, force: jnp.ndarray = None
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Calculate the macroscopic density and velocity fields from the population distribution.

        Args:
            f (jnp.ndarray): Population distribution, shape (nx, ny, q, 1)
            force (jnp.ndarray, optional): External force field, shape (nx, ny, 1, 2)

        Returns:
            tuple: (rho, u_eq, force_total)
                rho (jnp.ndarray): Density field, shape (nx, ny, 1, 1)
                u_eq (jnp.ndarray): Force-corrected velocity for equilibrium, shape (nx, ny, 1, 2)
                force_total (jnp.ndarray): Total force (interaction + external), shape (nx, ny, 1, 2)
        """
        rho, u = super().__call__(f, force=None)  # Pass None to avoid any correction

        # Calculate interaction force
        force_int = self.force_int(rho)
        # Total force calculation
        if force is not None:
            force_total = force + force_int
        else:
            force_total = force_int

        u_eq = u + force_total / (2 * rho)  # divide by rho for proper correction

        return rho, u_eq, force_total

    @partial(jit, static_argnums=(0,))
    def eos(self, rho):
        """Equation of state - extract 2D data for computation"""
        rho_2d = rho[:, :, 0, 0]  # Extract (nx, ny) from (nx, ny, 1, 1)
        eos_2d = (
            2
            * self.beta
            * (rho_2d - self.rho_l)
            * (rho_2d - self.rho_v)
            * (2 * rho_2d - self.rho_l - self.rho_v)
        )

        # Convert back to 4D format
        eos_4d = jnp.zeros_like(rho)
        eos_4d = eos_4d.at[:, :, 0, 0].set(eos_2d)
        return eos_4d

    @partial(jit, static_argnums=(0,))
    def chem_pot(self, rho):
        """
        Calculate the chemical potential.
        """
        mu_0 = self.eos(rho)
        chem_pot__ = mu_0 - self.kappa * self.laplacian(rho)
        return chem_pot__

    @partial(jit, static_argnums=(0,))
    def force_int(self, rho):
        """
        Calculate the interaction force. Note that the chemical potential uses the same gradient calculation as the
        density. This work because the values of the chemical potential are different to the rho_v and rho_l values.
        If these where to match it would fail, but that will often not happen.
        """
        grad_chem_pot = self.gradient.standard(self.chem_pot(rho), determine_padding_modes(self.bc_config))
        return -rho * grad_chem_pot

    @partial(jit, static_argnums=(0,))
    def u_new(self, u, force):
        """
        Update velocity with interaction force.
        """
        # Both u and force have shape (nx, ny, 1, 2)
        return u + force / 2




