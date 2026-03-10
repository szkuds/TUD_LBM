from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import jit

from .update import Update
from simulation_operators import MacroscopicMultiphaseCS
from app_setup.registry import register_operator, get_operators

if TYPE_CHECKING:
    from app_setup.simulation_setup import SimulationSetup


@register_operator("update_timestep")
class UpdateMultiphase(Update):
    name = "multiphase"
    """
    Performs the full multiphase LBM update_timestep step.

    Usage:
        UpdateMultiphase(app_setup=multiphase_config)
    """

    def __init__(self, config: "SimulationSetup") -> None:
        """
        Initialize the UpdateMultiphase operator.

        Args:
            config: SimulationSetup object containing all simulation_type parameters.
        """
        super().__init__(config)

        # Override macroscopic with multiphase version from registry
        eos = config.eos
        extra = config.extra if hasattr(config, 'extra') else {}

        macroscopic_ops = get_operators("macroscopic")
        if eos not in macroscopic_ops:
            available = ", ".join(sorted(macroscopic_ops.keys()))
            raise ValueError(
                f"Unknown EOS: '{eos}'. Available macroscopic operators: {available}"
            )

        if eos == "carnahan-starling":
            # MacroscopicMultiphaseCS has a different constructor signature
            self.macroscopic = MacroscopicMultiphaseCS(
                grid=self.grid,
                lattice=self.lattice,
                kappa=config.kappa,
                interface_width=config.interface_width,
                rho_l=config.rho_l,
                rho_v=config.rho_v,
                a_eos=extra['a_eos'],
                b_eos=extra['b_eos'],
                r_eos=extra['r_eos'],
                t_eos=extra['t_eos'],
                force_enabled=config.force_enabled,
                bc_config=config.bc_config,
            )
        else:
            # Use the registry to get the macroscopic operator class
            macroscopic_cls = macroscopic_ops[eos].cls
            self.macroscopic = macroscopic_cls(config)

    @partial(jit, static_argnums=(0,))
    def __call__(self, f: jnp.ndarray, force: jnp.ndarray = None):
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
