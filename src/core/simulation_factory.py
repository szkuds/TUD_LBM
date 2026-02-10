"""SimulationFactory — creates the right simulation from config."""

from typing import Any, Dict

from simulation import SinglePhaseSimulation, MultiphaseSimulation


class SimulationFactory:
    """Factory for creating simulation instances."""

    # Keys that are config-only and should not be passed to simulation constructors
    EXCLUDED_KEYS = frozenset({
        "simulation_type", "save_interval", "results_dir",
        "init_type", "init_dir", "skip_interval", "save_fields"
    })

    @staticmethod
    def create(config: Dict[str, Any]):
        """Create a simulation instance from a config dict."""
        simulation_type = config.get("simulation_type", "singlephase")
        sim_kwargs = {k: v for k, v in config.items() if k not in SimulationFactory.EXCLUDED_KEYS}

        if simulation_type == "single_phase":
            return SinglePhaseSimulation(**sim_kwargs)
        elif simulation_type == "multiphase":
            return MultiphaseSimulation(**sim_kwargs)
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")

    # Keep legacy method for backward compatibility
    @staticmethod
    def create_simulation(simulation_type: str, **kwargs):
        """Legacy factory method (deprecated, use create() instead)."""
        if simulation_type == "singlephase":
            return SinglePhaseSimulation(**kwargs)
        elif simulation_type == "multiphase":
            return MultiphaseSimulation(**kwargs)
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")
