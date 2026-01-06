from functools import partial

import jax.numpy as jnp
from jax import jit

from src.core.lattice.lattice import Lattice
from src.core.grid.grid import Grid
from .update import Update
from src.operators import MacroscopicMultiphaseDW, MacroscopicMultiphaseCS


class UpdateMultiphase(Update):
    def __init__(
            self,
            grid: Grid,
            lattice: Lattice,
            tau: float,
            kappa: float,
            interface_width: int,
            rho_l: float,
            rho_v: float,
            bc_config: dict = None,
            force_enabled: bool = False,
            collision_scheme: str = "bgk",
            eos: str = "double-well",
            k_diag=None,
            **kwargs
    ):
        super().__init__(
            grid,
            lattice,
            tau,
            bc_config,
            force_enabled=force_enabled,
            collision_scheme=collision_scheme,
            k_diag=k_diag,
            **kwargs
        )
        if eos == "double-well":
            self.macroscopic = MacroscopicMultiphaseDW(
                grid, lattice, kappa, interface_width, rho_l, rho_v, force_enabled=force_enabled, bc_config=bc_config
            )
        #TODO: Need to make sure that the maxwell contruction is done to get the correct starting values.
        elif eos == "carnahan-starling":
            macroscopic_args = dict(
                grid=grid,
                lattice=lattice,
                kappa=kappa,
                interface_width=interface_width,
                rho_l=rho_l,
                rho_v=rho_v,
                a_eos=kwargs['a_eos'],
                b_eos=kwargs['b_eos'],
                r_eos=kwargs['r_eos'],
                t_eos=kwargs['t_eos'],
                force_enabled=force_enabled,
                bc_config=bc_config,
            )
            self.macroscopic = MacroscopicMultiphaseCS(**macroscopic_args)

        else:
            raise ValueError(f"Unknown EOS: {eos}")

    @partial(jit, static_argnums=(0,))
    def __call__(self, f: jnp.array, force: jnp.ndarray = None):
        # If force_enabled and no force provided, use a simple constant force for testing
        if self.force_enabled and force is None:
            raise TypeError(
                "When the force is enabled an external force needs to be provided"
            )
        elif self.force_enabled:
            rho_prev, u, force_tot = self.macroscopic(f, force=force)
        else:
            rho_prev, u, force_tot = self.macroscopic(f)  # In this case the total force is only the interaction force
        feq = self.equilibrium(rho_prev, u)
        source = self.source_term(rho_prev, u, force_tot)
        fcol = self.collision(f, feq, source)
        fstream = self.streaming(fcol)
        if self.boundary_condition is not None:
            fbc = self.boundary_condition(fstream, fcol)
            return fbc
        else:
            return fstream
