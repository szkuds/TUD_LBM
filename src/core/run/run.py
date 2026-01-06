import numpy as np
import jax.numpy as jnp
import inspect

from src.core.simulation import SinglePhaseSimulation, MultiphaseSimulation
from src.operators import CompositeForce


class SimulationFactory:
    @staticmethod
    def create_simulation(simulation_type, **kwargs):
        if simulation_type == "singlephase":
            return SinglePhaseSimulation(**kwargs)
        elif simulation_type == "multiphase":
            return MultiphaseSimulation(**kwargs)
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")


class Run:
    """
    Main class to configure and run lattice Boltzmann simulations using the simulation factory. Note that the parameters
    stated below are placeholders in the case the user does not specify them.
    """
    def __init__(
        self,
        simulation_type="singlephase",
        *,
        save_interval=100,
        results_dir="~/TUD_LBM/results",
        init_type="standard",
        init_dir=None,
        skip_interval=0,
        collision=None,  # Accept collision as a kwarg
        simulation_name=None,  # Added simulation_name parameter
        **kwargs,
    ):
        # Accept either a string or a dict for collision
        if collision is not None:
            if isinstance(collision, str):
                collision_cfg = {"collision_scheme": collision}
            elif isinstance(collision, dict):
                collision_cfg = collision.copy()
            else:
                raise ValueError(
                    "collision must be either a string (for BGK) or dict (for MRT config)."
                )
            kwargs.update(collision_cfg)
        self.simulation = SimulationFactory.create_simulation(simulation_type, **kwargs)
        self.save_interval = save_interval
        self.skip_interval = skip_interval
        self.results_dir = results_dir
        self.init_type = init_type
        self.init_dir = init_dir
        # Auto-detect simulation name from calling function if not provided
        if simulation_name is None:
            frame = inspect.currentframe()
            try:
                caller_frame = frame.f_back
                while caller_frame:
                    func_name = caller_frame.f_code.co_name
                    if func_name != "<module>" and not func_name.startswith("_"):
                        simulation_name = func_name
                        break
                    caller_frame = caller_frame.f_back
            finally:
                del frame
        self.config = self._build_config(
            simulation_type=simulation_type,
            save_interval=save_interval,
            results_dir=results_dir,
            skip_interval=skip_interval,
            init_type=init_type,
            init_dir=init_dir,
            **kwargs,
        )
        from src.util import SimulationIO

        self.io_handler = SimulationIO(
            base_dir=results_dir, config=self.config, simulation_name=simulation_name
        )

    def _build_config(self, **kwargs):
        # Simple config builder
        return dict(**kwargs)

    def _save_data(self, it, f_prev, **kwargs):
        # Save data using the simulation's macroscopic operator
        force_ext = None
        h_prev = None
        if hasattr(self.simulation, "macroscopic"):
            macroscopic = self.simulation.macroscopic
            if self.config.get("force_enabled") and self.config.get("force_obj"):
                if self.simulation.force_obj.electric_present:
                    rho = jnp.sum(f_prev, axis=2, keepdims=True)
                    h_prev = kwargs.get('h_i')
                    force_ext = self.simulation.force_obj.compute_force(rho=rho,
                                                                        rho_l=self.config.get('rho_l'),
                                                                        rho_v=self.config.get('rho_l'),
                                                                        h_i=h_prev)
                    result = macroscopic(f_prev, force_ext)
                else:
                    rho = jnp.sum(f_prev, axis=2, keepdims=True)
                    force = CompositeForce(*self.config.get("force_obj"))
                    if self.config.get("simulation_type") == "multiphase":
                        force_ext = force.compute_force(
                            rho=rho, rho_l=self.config.get("rho_l"), rho_v=self.config.get("rho_v")
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
        self.io_handler.save_data_step(it, data_to_save)

    def run(self, *, verbose=True):
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
                'ElectricalForce')
            h_prev = electric_force.init_h()
        nt = getattr(self.simulation, "nt", 1000)
        if verbose:
            print(f"Starting LBM simulation with {nt} time steps...")
            print(
                f"Config -> Grid: {self.simulation.grid_shape}, Multiphase: {self.simulation.multiphase}, Wetting: {self.simulation.wetting_enabled}, Force: {self.simulation.force_enabled}"
            )
        for it in range(nt):
            if electric_present:
                f_prev, h_prev = self.simulation.run_timestep(f_prev, it, h_i=h_prev)
            else:
                f_prev = self.simulation.run_timestep(f_prev, it)
            if jnp.isnan(f_prev).any():
                print(f"NaN encountered at timestep {it}. Stopping simulation.")
                break
            # skip initial transients then save every `save_interval`
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
