from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import jit

from simulation_operators.force.source import SourceTerm
from simulation_operators.stream import Streaming
from simulation_operators.equilibrium.equilibrium_wb import EquilibriumWB
from simulation_operators.macroscopic.macroscopic import Macroscopic
from app_setup.registry import register_operator, get_operators

if TYPE_CHECKING:
    from app_setup.simulation_setup import SimulationSetup


@register_operator("update_timestep")
class Update(object):
    name = "single_phase"
    """
    Performs the full LBM update_timestep step (macroscopic, equilibrium, collision_models, streaming, BC).

    Usage:
        Update(app_setup=simulation_config)
    """

    def __init__(self, config: "SimulationSetup") -> None:
        """
        Initialize the Update operator.

        Args:
            config: Configuration object containing all simulation_type parameters.
        """
        from simulation_domain.grid import Grid
        from simulation_domain.lattice import Lattice

        grid = Grid(config.grid_shape)
        lattice = Lattice(config.lattice_type)

        self.grid = grid
        self.lattice = lattice
        self.tau = config.tau
        self.force_enabled = config.force_enabled

        # Extract optional params from extra
        extra = config.extra if hasattr(config, 'extra') else {}
        self.bubble = extra.get('bubble', False)
        self.rho_ref = extra.get('rho_ref', False)
        self.g = extra.get('g', False)

        # Create simulation_operators
        self.macroscopic = Macroscopic(config)
        self.equilibrium = EquilibriumWB(config)

        # Select collision_models scheme from registry
        collision_ops = get_operators("collision_models")
        try:
            collision_cls = collision_ops[config.collision_scheme].cls
        except KeyError as exc:
            available = ", ".join(sorted(collision_ops.keys()))
            raise ValueError(
                f"Unknown collision_models scheme '{config.collision_scheme}'. "
                f"Available: {available}"
            ) from exc
        self.collision = collision_cls(config)

        self.source_term = SourceTerm(config)
        self.streaming = Streaming(config)

        # Resolve boundary condition from the registry
        if config.bc_config is not None:
            bc_ops = get_operators("boundary_condition")
            bc_cls = bc_ops["standard"].cls
            self.boundary_condition = bc_cls(config)
        else:
            self.boundary_condition = None

    @partial(jit, static_argnums=(0,))
    def __call__(self, f: jnp.ndarray, force: jnp.ndarray = None):
        if self.force_enabled:
            rho, u, force_tot = self.macroscopic(f, force=force)

            # Calculate source term and pass it to collision_models
            feq = self.equilibrium(rho, u)
            source = self.source_term(rho, u, force_tot)
            fcol = self.collision(f, feq, source)
        else:
            rho, u = self.macroscopic(f)
            feq = self.equilibrium(rho, u)
            fcol = self.collision(f, feq)

        fstream = self.streaming(fcol)
        if self.boundary_condition is not None:
            fbc = self.boundary_condition(fstream, fcol)
            return fbc
        else:
            return fstream
