from functools import partial

from jax import jit
import jax.numpy as jnp

from .base import BaseSimulation
from src.core.update import UpdateMultiphase, UpdateMultiphaseHysteresis
from src.operators import Initialise, CompositeForce


class MultiphaseSimulation(BaseSimulation):
    def __init__(
        self,
        grid_shape,
        lattice_type="D2Q9",
        tau=1.0,
        nt=1000,
        kappa=0.1,
        rho_l=1.0,
        rho_v=0.1,
        interface_width=4,
        force_enabled=False,
        force_obj=None,
        bc_config=None,
        collision_scheme="bgk",
        k_diag=None,
        eos="double-well",
        **kwargs
    ):
        super().__init__(grid_shape, lattice_type, tau, nt)
        self.update = None
        self.initialise = None
        self.macroscopic = None
        self.eos = eos
        self.kappa = kappa
        self.rho_l = rho_l
        self.rho_v = rho_v
        self.interface_width = interface_width
        self.force_enabled = force_enabled
        self.force_obj = CompositeForce(*force_obj) if force_obj is not None else None
        self.bc_config = bc_config
        self.collision_scheme = collision_scheme
        self.k_diag = k_diag
        self.kwargs = kwargs
        self.bubble = kwargs.get('bubble', False)
        self.rho_ref = self.kwargs.get('rho_ref', False)
        self.g = self.kwargs.get('g', False)
        self.setup_operators()
        self.multiphase = True

    def setup_operators(self):
        self.wetting_enabled = any(bc_type == 'wetting' for bc_type in (self.bc_config or {}).values())
        if self.bubble:
            self.initialise = Initialise(self.grid, self.lattice, self.bubble, self.g, self.rho_ref)
        else:
            self.initialise = Initialise(self.grid, self.lattice, self.bubble)
        if self.bc_config and "hysteresis_params" in self.bc_config:
            self.update = UpdateMultiphaseHysteresis(
                self.grid, self.lattice, self.tau, self.kappa, self.interface_width,
                self.rho_l, self.rho_v, self.bc_config, self.force_enabled,
                collision_scheme=self.collision_scheme, eos=self.eos,
                k_diag=self.k_diag, **self.kwargs
            )
        else:
            self.update = UpdateMultiphase(
                self.grid, self.lattice, self.tau, self.kappa, self.interface_width,
                self.rho_l, self.rho_v, self.bc_config, self.force_enabled,
                collision_scheme=self.collision_scheme, eos=self.eos,
                k_diag=self.k_diag, **self.kwargs
            )
        self.macroscopic = self.update.macroscopic

    def initialize_fields(self, init_type="multiphase_droplet", *, init_dir=None):
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
        if self.force_enabled and self.force_obj and self.force_obj.electric_present:
            rho = jnp.sum(f_prev, axis=2, keepdims=True)
            h_prev = kwargs.get('h_i')
            force_ext = self.force_obj.compute_force(
                rho=rho,
                rho_l=self.rho_l,
                rho_v=self.rho_v,
                h_i=h_prev
            )
            f_next = self.update(f_prev, force=force_ext)
            electric_force = self.force_obj.get_component_by_name(
                self.force_obj.forces, 'ElectricalForce'
            )
            conductivity = electric_force.conductivity(
                rho, electric_force.conductivity_liquid,
                electric_force.conductivity_vapour
            )
            h_next = electric_force.update_h_i(h_prev, conductivity)
            return f_next, h_next

        elif self.force_enabled and self.force_obj:
            rho = jnp.sum(f_prev, axis=2, keepdims=True)
            force_ext = self.force_obj.compute_force(
                rho=rho,
                rho_l=self.rho_l,
                rho_v=self.rho_v
            )
            f_next = self.update(f_prev, force=force_ext)
            return f_next
        else:
            return self.update(f_prev)
        # TODO: Here I need to add the logic to update the electric field. Also need to add it to the single phase sim.
