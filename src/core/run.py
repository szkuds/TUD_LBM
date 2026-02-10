"""
Run class for LBM simulations.

Composes ConfigLoader, SimulationFactory, SimulationRunner, and SimulationIO.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config_loader import ConfigLoader
from .simulation_factory import SimulationFactory
from .simulation_runner import SimulationRunner


class Run:
    """
    User-friendly wrapper that composes Factory + Runner + IO.

    Usage:
        sim = Run("config.toml")
        sim.run(verbose=True)

    Or with kwargs:
        sim = Run(simulation_type="single_phase", nx=100, ny=100, ...)
        sim.run()
    """

    def __init__(
        self,
        config_path_or_kwargs: Union[str, Path, Dict[str, Any], None] = None,
        *,
        simulation_type: str = "single_phase",
        save_interval: int = 100,
        results_dir: str = "~/TUD_LBM/results",
        init_type: str = "standard",
        init_dir: Optional[str] = None,
        skip_interval: int = 0,
        collision=None,
        simulation_name: Optional[str] = None,
        save_fields: Optional[list] = None,
        **kwargs,
    ):
        # Load config from file or use provided kwargs
        if config_path_or_kwargs is not None and not isinstance(config_path_or_kwargs, dict):
            # Load from TOML file
            self.config = ConfigLoader.load(config_path_or_kwargs)
            # Apply overrides from kwargs
            self.config.update(kwargs)
        else:
            # Build config from kwargs
            if isinstance(config_path_or_kwargs, dict):
                kwargs.update(config_path_or_kwargs)
            kwargs = ConfigLoader.normalise_collision(collision, kwargs)
            self.config = ConfigLoader.build_config(
                simulation_type=simulation_type,
                save_interval=save_interval,
                results_dir=results_dir,
                init_type=init_type,
                init_dir=init_dir,
                skip_interval=skip_interval,
                save_fields=save_fields,
                **kwargs,
            )

        # Infer simulation name
        simulation_name = ConfigLoader.infer_simulation_name(simulation_name)

        # Create simulation via factory
        self.simulation = SimulationFactory.create(self.config)

        # Setup IO
        from util import SimulationIO
        self.io_handler = SimulationIO(
            base_dir=self.config.get("results_dir", results_dir),
            config=self.config,
            simulation_name=simulation_name,
        )

        # Create runner
        self.runner = SimulationRunner(self.simulation, self.io_handler, self.config)

    def run(self, *, verbose: bool = True):
        """Run the simulation."""
        self.runner.run(verbose=verbose)

