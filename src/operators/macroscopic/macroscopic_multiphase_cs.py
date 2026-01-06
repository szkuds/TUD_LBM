from functools import partial

import jax.numpy as jnp
from jax import jit

from src.core.lattice.lattice import Lattice
from src.core.grid.grid import Grid
from .macroscopic_multiphase_dw import MacroscopicMultiphaseDW


class MacroscopicMultiphaseCS(MacroscopicMultiphaseDW):
    """
    Calculates macroscopic variables for multiphase simulations.
    Inherits from MacroscopicMultiphaseDW and overrides EOS for Carnahan-Starling.
    """

    def __init__(
        self,
        grid: Grid,
        lattice: Lattice,
        kappa: float,
        interface_width: int,
        rho_l: float,
        rho_v: float,
        a_eos: float,
        b_eos: float,
        r_eos: float,
        t_eos: float,

        force_enabled: bool = False,
        bc_config: dict = None,
    ):
        super().__init__(
            grid, lattice, kappa, interface_width, rho_l, rho_v, force_enabled=force_enabled, bc_config=bc_config
        )
        self.a_eos = a_eos
        self.b_eos = b_eos
        self.R_eos = r_eos
        self.T_eos = t_eos

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
        rho, u, force_int = super().__call__(f, force=None)  # Pass None to avoid any correction

        # Total force calculation
        if force is not None:
            force_total = force + force_int
        else:
            force_total = force_int

        u_eq = u #+ force_total / (2 * rho)  # divide by rho for proper correction

        return rho, u_eq, force_total

    @partial(jit, static_argnums=(0,))
    def eos(self, rho):
        """
        Carnahan-Starling EOS.
        """
        rho_2d = rho[:, :, 0, 0]
        eos_2d = (
            -(2 * self.a_eos) * rho_2d
            + self.R_eos * self.T_eos * (1 + jnp.log(rho_2d))
            + ((16 * self.R_eos * self.T_eos * (self.b_eos * rho_2d - 12))
               / jnp.power((-4 + self.b_eos * rho_2d), 3))
        )
        #convert back to 4D format
        eos_4d = jnp.zeros_like(rho)
        eos_4d = eos_4d.at[:, :, 0, 0].set(eos_2d)
        return eos_4d
