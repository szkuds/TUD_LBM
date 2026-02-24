"""
Run class for LBM simulations.

Composes SimulationFactory, SimulationRunner, and SimulationIO.
"""

import inspect
from typing import Optional

from config import SimulationBundle
from .simulation_factory import SimulationFactory
from .simulation_runner import SimulationRunner


class Run:
    """
    Entry point for running LBM simulations.

    Usage:
        from config import SimulationBundle, SinglePhaseConfig, RunnerConfig
        from core import Run

        bundle = SimulationBundle(
            simulation=SinglePhaseConfig(
                grid_shape=(100, 100),
                tau=0.6,
                nt=10000,
            ),
            runner=RunnerConfig(save_interval=1000),
        )
        sim = Run(bundle)
        sim.run(verbose=True)

    Args:
        bundle: A SimulationBundle containing simulation and runner configs.
    """

    def __init__(self, bundle: SimulationBundle):
        if not isinstance(bundle, SimulationBundle):
            raise TypeError(
                f"Expected SimulationBundle, got {type(bundle).__name__}. "
                "Use SimulationBundle to configure your simulation."
            )

        self.bundle = bundle
        self.config = bundle.to_dict()

        # Infer simulation name if not provided
        simulation_name = bundle.runner.simulation_name or self._infer_simulation_name()

        # Create simulation via factory — pass the typed config directly
        self.simulation = SimulationFactory.create(bundle.simulation)

        # Setup IO
        from util import SimulationIO
        self.io_handler = SimulationIO(
            base_dir=bundle.runner.results_dir,
            config=self.config,
            simulation_name=simulation_name,
        )

        # Create runner
        self.runner = SimulationRunner(self.simulation, self.io_handler, bundle.runner)

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
