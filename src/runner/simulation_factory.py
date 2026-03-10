"""SimulationFactory — creates the right simulation_type from app_setup."""

from typing import TYPE_CHECKING

from app_setup.registry import get_operators

if TYPE_CHECKING:
    from app_setup.simulation_config import SimulationConfig


class SimulationFactory:
    """
    Factory for creating simulation_type instances from typed app_setup dataclasses.

    Uses the global operator registry to resolve simulation_type types.
    """

    @staticmethod
    def create(config: "SimulationConfig"):
        """
        Create a simulation_type instance from a typed app_setup dataclass.

        Args:
            config: A SinglePhaseConfig or MultiphaseConfig instance.

        Returns:
            A configured simulation_type instance.
        """
        from app_setup.simulation_config import SinglePhaseConfig, MultiphaseConfig

        # Ensure simulation_type classes are imported (triggers registration)
        import simulation_type  # noqa: F401

        if isinstance(config, SinglePhaseConfig):
            sim_type = "single_phase"
        elif isinstance(config, MultiphaseConfig):
            sim_type = "multiphase"
        else:
            raise TypeError(
                f"Expected SinglePhaseConfig or MultiphaseConfig, "
                f"got {type(config).__name__}"
            )

        sim_ops = get_operators("simulation_type")
        try:
            entry = sim_ops[sim_type]
        except KeyError as exc:
            available = ", ".join(sorted(sim_ops.keys()))
            raise ValueError(
                f"Unknown simulation_type type '{sim_type}'. Available: {available}"
            ) from exc
        return entry.cls(config)

