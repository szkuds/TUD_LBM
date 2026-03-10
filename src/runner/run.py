"""
Run class for LBM simulations.

Composes SimulationFactory, SimulationRunner, and SimulationIO.
"""

import inspect
from typing import Optional

from app_setup import SimulationSetup
from .simulation_factory import SimulationFactory
from .simulation_runner import SimulationRunner


class Run:
    """
    Entry point for running LBM simulations.

    Usage:
        from app_setup import SimulationSetup
        from runner import Run

        setup = SimulationSetup(
            grid_shape=(100, 100),
            tau=0.6,
            nt=10000,
            save_interval=1000,
        )
        sim = Run(setup)
        sim.run(verbose=True)

    Args:
        setup: A SimulationSetup containing all simulation parameters.
    """

    def __init__(self, setup: SimulationSetup):
        if not isinstance(setup, SimulationSetup):
            raise TypeError(
                f"Expected SimulationSetup, got {type(setup).__name__}. "
                "Use SimulationSetup to configure your simulation."
            )

        self.setup = setup
        self.config = setup.to_dict()

        # Infer simulation name if not provided
        simulation_name = setup.simulation_name or self._infer_simulation_name()

        # Create simulation via factory
        self.simulation = SimulationFactory.create(setup)

        # Setup IO
        from util import SimulationIO
        self.io_handler = SimulationIO(
            base_dir=setup.results_dir,
            config=self.config,
            simulation_name=simulation_name,
        )

        # Create runner
        self.runner = SimulationRunner(self.simulation, self.io_handler, setup)

    @staticmethod
    def _infer_simulation_name() -> Optional[str]:
        """Auto-detect simulation name from calling function via stack inspection."""
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back if frame else None
            while caller_frame:
                func_name = caller_frame.f_code.co_name
                if func_name != "<module>" and not func_name.startswith("_"):
                    return func_name
                caller_frame = caller_frame.f_back
        finally:
            del frame
        return None

    def run(self, *, verbose: bool = True):
        """Run the simulation."""
        self.runner.run(verbose=verbose)
