import json
import jax.numpy as jnp
from typing import Dict
import numpy as np
from datetime import datetime
import os
import logging
import sys


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle JAX arrays
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        # Handle custom force objects
        if hasattr(obj, "__class__") and hasattr(obj, "__dict__"):
            result = {
                "__class__": obj.__class__.__name__,
                "__module__": obj.__class__.__module__,
            }
            return result
        # Handle other numpy arrays if present
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


class SimulationIO:
    """
    Handles all I/O operations for the simulation, including logging and saving results.
    """

    def __init__(self, base_dir: str = "results", config: Dict = None, simulation_name: str = None):
        """
        Initializes the IO handler.

        Args:
            base_dir (str): The base directory to store simulation results.
            config (Dict, optional): A dictionary containing the simulation configuration to save.
            simulation_name (str, optional): Name of the simulation to include in the results directory.
        """
        self.base_dir = os.path.expanduser(base_dir)
        self.simulation_name = simulation_name
        self.run_dir = self._create_timestamped_directory()
        self.data_dir = os.path.join(self.run_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self._setup_logging()

        if config:
            self.save_config(config)

    def _setup_logging(self) -> None:
        """
        Configure root logger so everything printed to the console is
        also written to <run_dir>/simulation.log. Existing handlers are
        cleared to avoid duplicate lines when multiple simulations run
        in the same Python interpreter (e.g. test suites).
        """
        log_file = os.path.join(self.run_dir, "simulation.log")

        # 1. Build handlers
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(fmt)

        # 2. Reset & attach
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)  # stale handlers from previous runs
        root.setLevel(logging.INFO)
        root.addHandler(file_handler)
        root.addHandler(console_handler)

        # 3. Mirror *all* prints to the same log file
        class _Tee(object):
            def __init__(self, *streams):
                self._streams = streams

            def write(self, msg):
                [s.write(msg) for s in self._streams]

            def flush(self):
                [s.flush() for s in self._streams]

        logfile_stream = open(log_file, "a", buffering=1)  # line-buffered
        sys.stdout = _Tee(sys.__stdout__, logfile_stream)
        sys.stderr = _Tee(sys.__stderr__, logfile_stream)  # capture tracebacks too

    def _create_timestamped_directory(self) -> str:
        """Creates a unique, timestamped directory for a single simulation run."""
        timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        if self.simulation_name:
            rundir = os.path.join(self.base_dir, f"{timestamp}_{self.simulation_name}")
        else:
            rundir = os.path.join(self.base_dir, timestamp)
        os.makedirs(rundir, exist_ok=True)
        print(f"Created results directory: {rundir}")
        return rundir

    def save_config(self, config: Dict):
        """Saves the simulation configuration to a JSON file using CustomJSONEncoder."""
        config_path = os.path.join(self.run_dir, "config.json")

        # Rename boundary condition details if present (avoids duplication)
        if "bc_config" in config:
            config["boundary_conditions"] = config.pop("bc_config")

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4, cls=CustomJSONEncoder)
        print(f"Configuration saved to {config_path}")

    def save_data_step(self, iteration: int, data: Dict[str, np.ndarray]):
        """Saves the data for a single timestep to a compressed .npz file."""
        filename = os.path.join(self.data_dir, f"timestep_{iteration}.npz")
        np.savez(filename, **data)
