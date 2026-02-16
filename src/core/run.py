"""
Run class for LBM simulations.

Composes ConfigLoader, SimulationFactory, SimulationRunner, and SimulationIO.
"""

from pathlib import Path
from typing import Any, Dict, Union

from config import RUN_DEFAULTS
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

    Keyword Args:
        simulation_type: Type of simulation ("single_phase" or "multiphase")
        save_interval: Steps between saves (default: 100)
        results_dir: Output directory (default: "~/TUD_LBM/results")
        init_type: Initialization type (default: "standard")
        init_dir: Directory for restart files (default: None)
        skip_interval: Steps to skip on restart (default: 0)
        collision: Collision scheme - str or dict (default: None)
        simulation_name: Name for output folder (default: auto-detected)
        save_fields: Fields to save (default: None)
    """

    def __init__(
        self,
        config: Union[str, Path, Dict[str, Any], None] = None,
        **kwargs,
    ):
        # Merge defaults with provided kwargs
        opts = {**RUN_DEFAULTS, **kwargs}
        collision = opts.pop("collision")
        simulation_name = opts.pop("simulation_name")

        # Load config from file or use provided kwargs
        if config is not None and not isinstance(config, dict):
            # Load from TOML file
            self.config = ConfigLoader.load(config)
            self.config.update(opts)
        else:
            # Build config from kwargs
            if isinstance(config, dict):
                opts.update(config)
            opts = ConfigLoader.normalise_collision(collision, opts)
            self.config = ConfigLoader.build_config(**opts)

        # Infer simulation name
        simulation_name = ConfigLoader.infer_simulation_name(simulation_name)

        # Create simulation via factory
        self.simulation = SimulationFactory.create(self.config)

        # Setup IO
        from util import SimulationIO
        self.io_handler = SimulationIO(
            base_dir=self.config.get("results_dir", RUN_DEFAULTS["results_dir"]),
            config=self.config,
            simulation_name=simulation_name,
        )

        # Create runner
        self.runner = SimulationRunner(self.simulation, self.io_handler, self.config)

    def run(self, *, verbose: bool = True):
        """Run the simulation."""
        self.runner.run(verbose=verbose)

