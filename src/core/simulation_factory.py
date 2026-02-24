"""SimulationFactory — creates the right simulation from config."""

from typing import Any, Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import SinglePhaseSimulation, MultiphaseSimulation
    from config.simulation_config import SinglePhaseConfig, MultiphaseConfig, SimulationConfig


class SimulationFactory:
    """
    Factory for creating simulation instances from typed config dataclasses.
    """

    @staticmethod
    def create(config: "SimulationConfig"):
        """
        Create a simulation instance from a typed config dataclass.

        Args:
            config: A SinglePhaseConfig or MultiphaseConfig instance.

        Returns:
            A configured simulation instance.
        """
        from simulation import SinglePhaseSimulation, MultiphaseSimulation
        from config.simulation_config import SinglePhaseConfig, MultiphaseConfig

        if isinstance(config, SinglePhaseConfig):
            return SinglePhaseSimulation(config)
        elif isinstance(config, MultiphaseConfig):
            return MultiphaseSimulation(config)
        else:
            raise TypeError(
                f"Expected SinglePhaseConfig or MultiphaseConfig, "
                f"got {type(config).__name__}"
            )
