import json
import logging
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path
from types import MethodType
import jax.numpy as jnp
from .output_data import output_writers


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle JAX arrays
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        # Handle custom force objects
        if hasattr(obj, "__class__") and hasattr(obj, "__dict__"):
            return {
                "__class__": obj.__class__.__name__,
                "__module__": obj.__class__.__module__,
            }
        # Handle other numpy arrays if present
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


class SimulationIO:
    """Handles all I/O operations for the simulation, including logging and saving results."""

    def __init__(
        self,
        base_dir: str = "results",
        config: dict | None = None,
        simulation_name: str | None = None,
        output_format: str = "Numpy",
    ):
        """Initialises the IO handler.

        Args:
            base_dir (str): The base directory to store simulation results.
            config (Dict, optional): A dictionary containing the simulation configuration to save.
            simulation_name (str, optional): Name of the simulation to include in the results directory.
            output_format (str): Output writer format — ``"Numpy"`` (default) or ``"Vtk"``.
        """
        self.base_dir = str(Path(base_dir).expanduser())
        self.simulation_name = simulation_name
        self.run_dir = self._create_timestamped_directory()
        self.data_dir = str(Path(self.run_dir) / "data")
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        if config:
            self.save_config(config)

        self.save_data_step = MethodType(
            output_writers[output_format].save_data_step,
            self,
        )

    def _setup_logging(self) -> None:
        """Configure root logger so everything printed to the console is
        also written to <run_dir>/simulation.log. Existing handlers are
        cleared to avoid duplicate lines when multiple simulations run
        in the same Python interpreter (e.g. test suites).
        """
        log_file = str(Path(self.run_dir) / "simulation.log")

        # 1. Build handlers
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
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
        class _Tee:
            def __init__(self, *streams):
                self._streams = streams

            def write(self, msg):
                [s.write(msg) for s in self._streams]

            def flush(self):
                [s.flush() for s in self._streams]

        logfile_stream = Path(log_file).open("a", buffering=1)  # noqa: SIM115
        sys.stdout = _Tee(sys.__stdout__, logfile_stream)
        sys.stderr = _Tee(sys.__stderr__, logfile_stream)  # capture tracebacks too

    def _create_timestamped_directory(self) -> str:
        """Creates a unique, timestamped directory for a single simulation run."""
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d/%H-%M-%S")
        base = Path(self.base_dir)
        suffix = f"{timestamp}_{self.simulation_name}" if self.simulation_name else timestamp
        run_dir = base / suffix
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created results directory: {run_dir}")
        return str(run_dir)

    def save_config(self, config: dict):
        """Saves the simulation configuration to a JSON file using CustomJSONEncoder."""
        config_path = Path(self.run_dir) / "config.json"

        # Rename boundary condition details if present (avoids duplication)
        if "bc_config" in config:
            config["boundary_conditions"] = config.pop("bc_config")

        with config_path.open("w") as f:
            json.dump(config, f, indent=4, cls=CustomJSONEncoder)
        print(f"Configuration saved to {config_path}")
