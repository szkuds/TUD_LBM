"""SimulationFactory — creates the right simulation from a SimulationSetup."""

from typing import TYPE_CHECKING

from app_setup.registry import get_operators

if TYPE_CHECKING:
    from app_setup.simulation_setup import SimulationSetup


class SimulationFactory:
    """
    Factory for creating simulation instances from a SimulationSetup.

    Uses the global operator registry to resolve simulation types.
    """

    @staticmethod
    def create(setup: "SimulationSetup"):
        """
        Create a simulation instance from a SimulationSetup.

        Args:
            setup: A SimulationSetup instance.

        Returns:
            A configured simulation instance.
        """
        # Ensure simulation classes are imported (triggers registration)
        import simulation_type  # noqa: F401

        sim_type = setup.sim_type

        sim_ops = get_operators("simulation_type")
        try:
            entry = sim_ops[sim_type]
        except KeyError as exc:
            available = ", ".join(sorted(sim_ops.keys()))
            raise ValueError(
                f"Unknown simulation type '{sim_type}'. Available: {available}"
            ) from exc
        return entry.cls(setup)
