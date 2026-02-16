"""SimulationFactory — creates the right simulation from config."""

from typing import Any, Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import SinglePhaseSimulation, MultiphaseSimulation
    from config.simulation_config import SinglePhaseConfig, MultiphaseConfig


class SimulationFactory:
    """
    Factory for creating simulation instances.

    Handles the translation from raw dict/kwargs to typed config dataclasses.
    """

    # Keys that are runner-only and should not be passed to simulation configs
    RUNNER_KEYS = frozenset({
        "simulation_type", "save_interval", "results_dir",
        "init_type", "init_dir", "skip_interval", "save_fields"
    })

    @staticmethod
    def _build_single_phase_config(kwargs: Dict[str, Any]) -> "SinglePhaseConfig":
        """Build SinglePhaseConfig from kwargs dict."""
        from config.simulation_config import SinglePhaseConfig

        # Extract known keys, put rest in extra
        known = {"grid_shape", "lattice_type", "tau", "nt", "force_enabled",
                 "force_obj", "bc_config", "collision_scheme", "k_diag"}
        config_kwargs = {k: v for k, v in kwargs.items() if k in known}
        extra = {k: v for k, v in kwargs.items() if k not in known}

        # Ensure grid_shape is tuple
        if "grid_shape" in config_kwargs and not isinstance(config_kwargs["grid_shape"], tuple):
            config_kwargs["grid_shape"] = tuple(config_kwargs["grid_shape"])

        return SinglePhaseConfig(**config_kwargs, extra=extra)

    @staticmethod
    def _build_multiphase_config(kwargs: Dict[str, Any]) -> "MultiphaseConfig":
        """Build MultiphaseConfig from kwargs dict."""
        from config.simulation_config import MultiphaseConfig

        # Extract known keys, put rest in extra
        known = {"grid_shape", "lattice_type", "tau", "nt", "kappa", "rho_l", "rho_v",
                 "interface_width", "eos", "force_enabled", "force_obj", "bc_config",
                 "collision_scheme", "k_diag", "bubble", "rho_ref", "g"}
        config_kwargs = {k: v for k, v in kwargs.items() if k in known}
        extra = {k: v for k, v in kwargs.items() if k not in known}

        # Ensure grid_shape is tuple
        if "grid_shape" in config_kwargs and not isinstance(config_kwargs["grid_shape"], tuple):
            config_kwargs["grid_shape"] = tuple(config_kwargs["grid_shape"])

        return MultiphaseConfig(**config_kwargs, extra=extra)

    @staticmethod
    def create(config: Union[Dict[str, Any], "SinglePhaseConfig", "MultiphaseConfig"]):
        """
        Create a simulation instance from a config dict or config dataclass.

        Args:
            config: Either a dict with simulation parameters, or a
                    SinglePhaseConfig/MultiphaseConfig dataclass instance.

        Returns:
            A configured simulation instance.
        """
        from simulation import SinglePhaseSimulation, MultiphaseSimulation
        from config.simulation_config import SinglePhaseConfig, MultiphaseConfig

        # Handle config dataclass directly
        if isinstance(config, SinglePhaseConfig):
            return SinglePhaseSimulation(config)
        elif isinstance(config, MultiphaseConfig):
            return MultiphaseSimulation(config)

        # Handle dict config - build typed config first
        simulation_type = config.get("simulation_type", "single_phase")
        sim_kwargs = {k: v for k, v in config.items() if k not in SimulationFactory.RUNNER_KEYS}

        if simulation_type == "single_phase":
            cfg = SimulationFactory._build_single_phase_config(sim_kwargs)
            return SinglePhaseSimulation(cfg)
        elif simulation_type == "multiphase":
            cfg = SimulationFactory._build_multiphase_config(sim_kwargs)
            return MultiphaseSimulation(cfg)
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")
