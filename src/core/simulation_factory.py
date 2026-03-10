"""SimulationFactory — creates the right simulation from config."""

from typing import TYPE_CHECKING

from registry import get_operators

if TYPE_CHECKING:
    from config.simulation_config import SimulationConfig


class SimulationFactory:
    """
    Factory for creating simulation instances from typed config dataclasses.

    Uses the global operator registry to resolve simulation types.
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
        from config.simulation_config import SinglePhaseConfig, MultiphaseConfig

        # Ensure simulation classes are imported (triggers registration)
        import simulation  # noqa: F401

        if isinstance(config, SinglePhaseConfig):
            sim_type = "single_phase"
        elif isinstance(config, MultiphaseConfig):
            sim_type = "multiphase"
        else:
            raise TypeError(
                f"Expected SinglePhaseConfig or MultiphaseConfig, "
                f"got {type(config).__name__}"
            )

        sim_ops = get_operators("simulation")
        try:
            entry = sim_ops[sim_type]
        except KeyError as exc:
            available = ", ".join(sorted(sim_ops.keys()))
            raise ValueError(
                f"Unknown simulation type '{sim_type}'. Available: {available}"
            ) from exc
        return entry.cls(config)

