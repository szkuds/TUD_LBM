from typing import Dict
import numpy as np
from datetime import datetime
import os
import logging
import sys
import tomli_w

from config import BASE_RESULTS_DIR


# Constants for TOML config file structure (used when saving config)
_TOML_SIMULATION_KEYS = [
    "simulation_type",
    "grid_shape",
    "lattice_type",
    "tau",
    "nt",
    "save_interval",
    "skip_interval",
    "init_type",
    "init_dir",
    "simulation_name",
]

_TOML_MULTIPHASE_KEYS = [
    "kappa",
    "rho_l",
    "rho_v",
    "interface_width",
    "eos",
]

_TOML_OUTPUT_KEYS = [
    "results_dir",
    "save_fields",
]


class SimulationIO:
    """
    Handles all I/O operations for the simulation, including logging and saving results.
    """

    def __init__(self, base_dir: str = BASE_RESULTS_DIR, config: Dict = None, simulation_name: str = None):
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
            run_dir = os.path.join(self.base_dir, f"{timestamp}_{self.simulation_name}")
        else:
            run_dir = os.path.join(self.base_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Created results directory: {run_dir}")
        return run_dir

    def save_config(self, config: Dict):
        """Saves the simulation configuration to a TOML file."""
        # Rename boundary condition details if present (avoids duplication)
        if "bc_config" in config:
            config["boundary_conditions"] = config.pop("bc_config")

        # Save TOML
        toml_config = self._restructure_for_toml(config)
        toml_path = os.path.join(self.run_dir, "config.toml")

        with open(toml_path, "wb") as f:
            tomli_w.dump(toml_config, f)
        print(f"Configuration saved to {toml_path}")

    def _restructure_for_toml(self, config: Dict) -> Dict:
        """Restructure flat config dict into TOML sections.

        Args:
            config: Flat configuration dictionary

        Returns:
            Structured dictionary suitable for TOML serialization
        """
        toml_cfg = {}

        def sanitize_for_toml(obj):
            """Recursively sanitize an object for TOML serialization."""
            if obj is None:
                return None  # Will be filtered out
            elif isinstance(obj, dict):
                # Recursively sanitize dict, filtering out None values
                return {k: sanitize_for_toml(v) for k, v in obj.items()
                        if v is not None and sanitize_for_toml(v) is not None}
            elif isinstance(obj, (list, tuple)):
                # Convert to list and sanitize elements, filter out None
                return [sanitize_for_toml(item) for item in obj if item is not None]
            elif hasattr(obj, '__class__') and not isinstance(obj, (str, int, float, bool)):
                # For custom objects (like force objects), just store class name
                return f"<{obj.__class__.__name__}>"
            else:
                return obj

        # Simulation section
        simulation = {}
        for key in _TOML_SIMULATION_KEYS:
            if key in config and config[key] is not None:
                val = config[key]
                # Convert tuples to lists for TOML
                if isinstance(val, tuple):
                    val = list(val)
                if key == "simulation_type":
                    simulation["type"] = val
                else:
                    simulation[key] = val
        if simulation:
            toml_cfg["simulation"] = simulation

        # Multiphase section
        multiphase = {}
        for key in _TOML_MULTIPHASE_KEYS:
            if key in config and config[key] is not None:
                multiphase[key] = config[key]
        if multiphase:
            toml_cfg["multiphase"] = multiphase

        # Boundary conditions section - sanitize nested dict
        if "boundary_conditions" in config and config["boundary_conditions"] is not None:
            bc_sanitized = sanitize_for_toml(config["boundary_conditions"])
            if bc_sanitized:
                toml_cfg["boundary_conditions"] = bc_sanitized

        # Output section
        output = {}
        for key in _TOML_OUTPUT_KEYS:
            if key in config and config[key] is not None:
                output[key] = config[key]
        if output:
            toml_cfg["output"] = output

        return toml_cfg

    def save_data_step(self, iteration: int, data: Dict[str, np.ndarray]):
        """Saves the data for a single timestep to a compressed .npz file."""
        filename = os.path.join(self.data_dir, f"timestep_{iteration}.npz")
        np.savez(filename, **data)
