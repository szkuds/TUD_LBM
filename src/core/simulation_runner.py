"""SimulationRunner — owns the time-loop, data saving, and NaN-checking."""

import numpy as np
import jax.numpy as jnp
from typing import Any, Dict

from .step_result import StepResult


class SimulationRunner:
    """
    Owns:
    - time stepping loop
    - NaN checking
    - saving
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

    def _save_data(self, it: int, step_result: StepResult):
        """Save data from the StepResult."""
        data_to_save = {
            "f": np.array(step_result.f),
        }

        if step_result.rho is not None:
            data_to_save["rho"] = np.array(step_result.rho)
        if step_result.u is not None:
            data_to_save["u"] = np.array(step_result.u)
        if step_result.force is not None:
            data_to_save["force"] = np.array(step_result.force)
        if step_result.force_ext is not None:
            data_to_save["force_ext"] = np.array(step_result.force_ext)
        if step_result.h is not None:
            data_to_save["h"] = np.array(step_result.h)

        # Filter data_to_save if save_fields is specified
        if self.save_fields is not None:
            data_to_save = {
                k: v for k, v in data_to_save.items()
                if k in self.save_fields
            }

        self.io_handler.save_data_step(it, data_to_save)

    def run(self, *, verbose=True):
        """Run the simulation time loop. Only iterates and delegates."""
        f_prev = self.simulation.initialise_fields(
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
            # Run timestep - returns StepResult
            step_result = self.simulation.run_timestep(f_prev, it, h_i=h_prev)

            # Extract f and h for next iteration
            f_prev = step_result.f
            if step_result.h is not None:
                h_prev = step_result.h

            if jnp.isnan(f_prev).any():
                print(f"NaN encountered at timestep {it}. Stopping simulation.")
                break

            # Skip initial transients then save every `save_interval`
            if (it > self.skip_interval) and (
                it % self.save_interval == 0 or it == nt - 1
            ):
                self._save_data(it, step_result)

                if verbose and step_result.rho is not None and step_result.u is not None:
                    rho = step_result.rho
                    u = step_result.u
                    avg_rho = np.mean(rho)
                    max_u = np.max(np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2))
                    print(
                        f"Step {it}/{nt}: avg_rho={avg_rho:.4f}, max_u={max_u:.6f}"
                    )

        if verbose:
            print("Simulation completed!")
            print(f"Results saved in: {self.io_handler.run_dir}")
