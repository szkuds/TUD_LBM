from functools import partial

from jax import jit
import jax.numpy as jnp

from .base import BaseSimulation
from update_timestep import UpdateMultiphase, UpdateMultiphaseHysteresis
from simulation_operators import Initialise, CompositeForce
from runner.step_result import StepResult
from app_setup.simulation_config import MultiphaseConfig
from app_setup.registry import register_operator


@register_operator("simulation_type")
class MultiphaseSimulation(BaseSimulation):
    name = "multiphase"
    """
    Multiphase (two-phase) LBM simulation_type.

    Args:
        app_setup: A MultiphaseConfig dataclass with all simulation_type parameters.
    """

    def __init__(self, config: MultiphaseConfig):
        if not isinstance(config, MultiphaseConfig):
            raise TypeError(f"app_setup must be MultiphaseConfig, got {type(config)}")

        super().__init__(config.grid_shape, config.lattice_type, config.tau, config.nt)

        # Store app_setup and extract frequently-used attributes
        self.config = config
        self.eos = config.eos
        self.kappa = config.kappa
        self.rho_l = config.rho_l
        self.rho_v = config.rho_v
        self.interface_width = config.interface_width
        self.force_enabled = config.force_enabled
        self.force_obj = CompositeForce(*config.force_obj) if config.force_obj else None
        self.bc_config = config.bc_config
        self.collision_scheme = config.collision_scheme
        self.k_diag = config.k_diag
        self.bubble = config.bubble
        self.rho_ref = config.rho_ref
        self.g = config.g
        self.optional = config.extra

        self.update = None
        self.initialise = None
        self.macroscopic = None
        self.multiphase = True
        self.setup_operators()

    def setup_operators(self):
        self.wetting_enabled = any(bc_type == 'wetting' for bc_type in (self.bc_config or {}).values())
        self.initialise = Initialise(self.config)
        if self.bc_config and "hysteresis_params" in self.bc_config:
            self.update = UpdateMultiphaseHysteresis(self.config)
        else:
            self.update = UpdateMultiphase(self.config)
        self.macroscopic = self.update.macroscopic

    def initialise_fields(self, init_type="multiphase_droplet", *, init_dir=None):
        if init_type == "init_from_file":
            if init_dir is None:
                raise ValueError(
                    "init_from_file requires init_dir pointing to a .npz file"
                )
            return self.initialise.init_from_npz(init_dir)

        elif init_type == "multiphase_droplet":
            return self.initialise.initialise_multiphase_droplet(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_bubble":
            return self.initialise.initialise_multiphase_bubble(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_droplet_top":
            return self.initialise.initialise_multiphase_droplet_top(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_bubble_bot":
            return self.initialise.initialise_multiphase_bubble_bot(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_bubble_bubble":
            return self.initialise.initialise_multiphase_bubble_bubble(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_lateral_bubble_configuration":
            return self.initialise.initialise_multiphase_lateral_bubble_configuration(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "wetting_chem_step":
            return self.initialise.initialise_wetting_chemical_step(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "wetting":
            return self.initialise.initialise_wetting(
                self.rho_l, self.rho_v, self.interface_width
            )
        else:
            return self.initialise.initialise_standard()

    @partial(jit, static_argnums=(0,))
    def run_timestep(self, f_prev, it, **kwargs):
        h_prev = kwargs.get('h_i')
        h_next = None
        force_ext = None

        if self.force_enabled and self.force_obj and self.force_obj.electric_present:
            rho = jnp.sum(f_prev, axis=2, keepdims=True)
            force_ext = self.force_obj.compute_force(
                rho=rho,
                rho_l=self.rho_l,
                rho_v=self.rho_v,
                h_i=h_prev
            )
            f_next = self.update(f_prev, force=force_ext)
            electric_force = self.force_obj.get_component_by_name(
                self.force_obj.forces, 'electric'
            )
            conductivity = electric_force.conductivity(
                rho, electric_force.conductivity_liquid,
                electric_force.conductivity_vapour
            )
            h_next = electric_force.update_h_i(h_prev, conductivity)

        elif self.force_enabled and self.force_obj:
            rho = jnp.sum(f_prev, axis=2, keepdims=True)
            force_ext = self.force_obj.compute_force(
                rho=rho,
                rho_l=self.rho_l,
                rho_v=self.rho_v
            )
            f_next = self.update(f_prev, force=force_ext)
        else:
            f_next = self.update(f_prev)

        # Compute macroscopic fields for StepResult
        result = self.macroscopic(f_next, force_ext) if force_ext is not None else self.macroscopic(f_next)

        if isinstance(result, tuple) and len(result) == 3:
            rho, u, force = result
            return StepResult(f=f_next, rho=rho, u=u, force=force, force_ext=force_ext, h=h_next)
        else:
            rho, u = result
            return StepResult(f=f_next, rho=rho, u=u, force_ext=force_ext, h=h_next)
        # TODO: Here I need to add the logic to update_timestep the electric field. Also need to add it to the single phase sim.
