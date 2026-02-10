"""SimulationRunner — owns the time-loop, data saving, and NaN-checking."""

import numpy as np
import jax.numpy as jnp
from typing import Any, Dict

from operators import CompositeForce


class SimulationRunner:
    """
    Owns:
    - time stepping loop
    - NaN checking
    - save cadence
    Delegates:
    - field initialisation and per-step update to the simulation
    - persistence to io_handler
    """

    def __init__(
        self,
        simulation,
        io_handler,
        config: Dict[str, Any],
    ):
        self.simulation = simulation
        self.io_handler = io_handler
        self.config = config
        self.init_type = config.get("init_type", "standard")
        self.init_dir = config.get("init_dir")
        self.save_interval = int(config.get("save_interval", 100))
        self.skip_interval = int(config.get("skip_interval", 0))
        self.save_fields = config.get("save_fields")

    def _save_data(self, it, f_prev, **kwargs):
        """Save data using the simulation's macroscopic operator."""
        force_ext = None
        h_prev = None
        if hasattr(self.simulation, "macroscopic"):
            macroscopic = self.simulation.macroscopic
            if self.config.get("force_enabled") and self.config.get("force_obj"):
                if self.simulation.force_obj.electric_present:
                    rho = jnp.sum(f_prev, axis=2, keepdims=True)
                    h_prev = kwargs.get('h_i')
                    force_ext = self.simulation.force_obj.compute_force(
                        rho=rho,
                        rho_l=self.config.get('rho_l'),
                        rho_v=self.config.get('rho_l'),
                        h_i=h_prev
                    )
                    result = macroscopic(f_prev, force_ext)
                else:
                    rho = jnp.sum(f_prev, axis=2, keepdims=True)
                    force = CompositeForce(*self.config.get("force_obj"))
                    if self.config.get("simulation_type") == "multiphase":
                        force_ext = force.compute_force(
                            rho=rho,
                            rho_l=self.config.get("rho_l"),
                            rho_v=self.config.get("rho_v")
                        )
                    else:
                        force_ext = force.compute_force(rho)
                    result = macroscopic(f_prev, force_ext)
            else:
                result = macroscopic(f_prev)

            if isinstance(result, tuple) and len(result) == 3:
                rho, u, force = result
                data_to_save = {
                    "rho": np.array(rho),
                    "u": np.array(u),
                    "force": np.array(force),
                    "force_ext": np.array(force_ext),
                    "f": np.array(f_prev),
                    "h": np.array(h_prev)
                }
            else:
                rho, u = result
                data_to_save = {
                    "rho": np.array(rho),
                    "u": np.array(u),
                    "f": np.array(f_prev),
                }
        else:
            data_to_save = {"f": np.array(f_prev)}

        # Filter data_to_save if save_fields is specified
        if self.save_fields is not None:
            data_to_save = {
                k: v for k, v in data_to_save.items()
                if k in self.save_fields and v is not None
            }

        self.io_handler.save_data_step(it, data_to_save)

    def run(self, *, verbose=True):
        """Run the simulation time loop. Only iterates and delegates."""
        f_prev = self.simulation.initialize_fields(
            self.init_type, init_dir=self.init_dir
        )
        h_prev = None
        electric_present = False

        if self.simulation.force_enabled:
            electric_present = self.simulation.force_obj.electric_present

        if electric_present:
            electric_force = self.simulation.force_obj.get_component_by_name(
                self.simulation.force_obj.forces,
                'ElectricalForce'
            )
            h_prev = electric_force.init_h()

        nt = getattr(self.simulation, "nt", 1000)

        if verbose:
            print(f"Starting LBM simulation with {nt} time steps...")
            print(
                f"Config -> Grid: {self.simulation.grid_shape}, "
                f"Multiphase: {self.simulation.multiphase}, "
                f"Wetting: {self.simulation.wetting_enabled}, "
                f"Force: {self.simulation.force_enabled}"
            )

        for it in range(nt):
            if electric_present:
                f_prev, h_prev = self.simulation.run_timestep(f_prev, it, h_i=h_prev)
            else:
                f_prev = self.simulation.run_timestep(f_prev, it)

            if jnp.isnan(f_prev).any():
                print(f"NaN encountered at timestep {it}. Stopping simulation.")
                break

            # Skip initial transients then save every `save_interval`
            if (it > self.skip_interval) and (
                it % self.save_interval == 0 or it == nt - 1
            ):
                if electric_present:
                    self._save_data(it, f_prev, h_i=h_prev)
                else:
                    self._save_data(it, f_prev)

                if verbose and hasattr(self.simulation, "macroscopic"):
                    result = self.simulation.macroscopic(f_prev)
                    if isinstance(result, tuple) and len(result) >= 2:
                        rho, u = result[:2]
                        avg_rho = np.mean(rho)
                        max_u = np.max(np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2))
                        print(
                            f"Step {it}/{nt}: avg_rho={avg_rho:.4f}, max_u={max_u:.6f}"
                        )

        if verbose:
            print("Simulation completed!")
            print(f"Results saved in: {self.io_handler.run_dir}")
