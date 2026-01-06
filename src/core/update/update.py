from functools import partial

import jax.numpy as jnp
from jax import jit

from src.core.grid.grid import Grid
from src.core.lattice.lattice import Lattice
from src.core.collision import CollisionBGK, CollisionMRT, SourceTerm
from src.core.stream import Streaming
from src.operators.boundary_condition.boundary_condition import BoundaryCondition
from src.operators.equilibrium.equilibrium import Equilibrium
from src.operators.macroscopic.macroscopic import Macroscopic


class Update(object):
    def __init__(
        self,
        grid: Grid,
        lattice: Lattice,
        tau: float,
        bc_config: dict = None,
        force_enabled: bool = False,
        collision_scheme: str = "bgk",
        k_diag=None,
        **kwargs
    ):
        self.grid = grid
        self.lattice = lattice
        self.bubble = kwargs.get('bubble', False)
        self.rho_ref = kwargs.get('rho_ref', False)
        self.g = kwargs.get('g', False)
        self.tau = tau
        self.macroscopic = Macroscopic(grid, lattice, force_enabled=force_enabled)
        self.equilibrium = Equilibrium(self.grid, self.lattice)
        # Select collision scheme
        if collision_scheme == "mrt":
            # Extract MRT parameters from kwargs if provided
            mrt_params = {}
            for param in ["k0", "kb", "k2", "k4", "kv"]:
                if param in kwargs:
                    mrt_params[param] = kwargs[param]
            self.collision = CollisionMRT(grid, lattice, k_diag=k_diag, **mrt_params)
        else:
            self.collision = CollisionBGK(grid, lattice, tau)
        self.source_term = SourceTerm(grid, lattice, bc_config)
        self.streaming = Streaming(lattice)
        if bc_config is not None:
            self.boundary_condition = BoundaryCondition(grid, lattice, bc_config)
        else:
            self.boundary_condition = None
        self.force_enabled = force_enabled

    @partial(jit, static_argnums=(0,))
    def __call__(self, f: jnp.ndarray, force: jnp.ndarray = None):
        if self.force_enabled:
            rho, u, force_tot = self.macroscopic(f, force=force)

            # Calculate source term and pass it to collision
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
