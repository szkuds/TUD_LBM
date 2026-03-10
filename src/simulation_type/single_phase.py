import jax.numpy as jnp

from .base import BaseSimulation
from update_timestep import Update
from runner.step_result import StepResult
from app_setup.simulation_setup import SimulationSetup
from app_setup.registry import register_operator


@register_operator("simulation_type")
class SinglePhaseSimulation(BaseSimulation):
    name = "single_phase"
    """
    Single-phase LBM simulation.

    Args:
        config: A SimulationSetup with all simulation parameters.
    """

    def __init__(self, config: SimulationSetup):
        if not isinstance(config, SimulationSetup):
            raise TypeError(f"config must be SimulationSetup, got {type(config)}")

        super().__init__(config.grid_shape, config.lattice_type, config.tau, config.nt)

        # Store app_setup and extract frequently-used attributes
        self.config = config
        self.force_enabled = config.force_enabled
        self.force_obj = config.force_obj
        self.bc_config = config.bc_config
        self.collision_scheme = config.collision_scheme
        self.k_diag = config.k_diag
        self.optional = config.extra

        self.macroscopic = None
        self.initialiser = None
        self.update = None
        self.boundary_condition = None
        self.multiphase = False
        self.wetting_enabled = False
        self.setup_operators()

    def setup_operators(self):
        self.update = Update(self.config)
        self.macroscopic = self.update.macroscopic
        self.boundary_condition = self._make_boundary_condition(self.config)

    def initialise_fields(self, init_type="standard", *, init_dir=None):
        initialiser = self._make_initialiser(init_type, self.config)
        if init_type == "init_from_file":
            if init_dir is None:
                raise ValueError(
                    "init_from_file requires init_dir pointing to a .npz file"
                )
            return initialiser(npz_path=init_dir)
        return initialiser()

    # TODO: Need to implement the changed electric field logic
    def run_timestep(self, f_prev, it, **kwargs):
        force_ext = None
        if self.force_enabled and self.force_obj:
            rho = jnp.sum(f_prev, axis=2, keepdims=True)
            force_ext = self.force_obj.compute_force(rho=rho)
        f_next = (
            self.update(f_prev, force=force_ext) if self.force_enabled
            else self.update(f_prev)
        )

        # Compute macroscopic fields for StepResult
        result = self.macroscopic(f_next, force_ext) if force_ext is not None else self.macroscopic(f_next)

        if isinstance(result, tuple) and len(result) == 3:
            rho, u, force = result
            return StepResult(f=f_next, rho=rho, u=u, force=force, force_ext=force_ext)
        else:
            rho, u = result
            return StepResult(f=f_next, rho=rho, u=u, force_ext=force_ext)
