"""Input/Output handler for simulation management and logging."""

import logging
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from tud_lbm.config.simulation_config import SimulationConfig
from .output_data import output_writers


class SimulationIO:
    """Handles all I/O operations for the simulation, including logging and saving results."""

    def __init__(
        self,
        base_dir: str = "results",
        config: SimulationConfig | None = None,
        simulation_name: str | None = None,
        output_format: str = "numpy",
        config_file_type: str = ".toml",
    ):
        """Initialises the IO handler.

        Args:
            base_dir (str): The base directory to store simulation results.
            config (SimulationConfig, optional): A SimulationConfig object
                containing the simulation configuration to save.
            simulation_name (str, optional): Name of the simulation to include in the results directory.
            output_format (str): Output writer format — ``"Numpy"`` (default) or ``"Vtk"``.
            config_file_type (str): Extension for the saved config file — ``".toml"`` (default).
                Must match a registered adapter in :func:`~config.adapter_base.get_adapter`.
        """
        self.base_dir = str(Path(base_dir).expanduser())
        self.simulation_name = simulation_name
        self.config_file_type = config_file_type
        self.run_dir = self._create_timestamped_directory()
        self.data_dir = str(Path(self.run_dir) / "data")
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        if config:
            self.save_config(config)

        self.save_data_step = output_writers[output_format].save_data_step.__get__(self, type(self))

    def _setup_logging(self) -> None:
        """Configure root logger so everything printed to the console is also written to <run_dir>/simulation.log.

        Existing handlers are cleared to avoid duplicate lines when multiple simulations run
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
            def __init__(self, *streams: Any) -> None:  # noqa: ANN401
                self._streams = streams

            def write(self, msg: str) -> None:
                [s.write(msg) for s in self._streams]

            def flush(self) -> None:
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
        return str(run_dir)

    def save_config(self, config: SimulationConfig) -> None:
        """Save the simulation configuration to the run directory.

        Accepts  a :class:`SimulationConfig`.
        Uses :func:`~config.adapter_base.get_adapter` to dispatch to the
        correct adapter based on :attr:`config_file_type`.

        Args:
            config: A configuration dict or SimulationConfig object.
        """
        from tud_lbm.config.adapter_base import get_adapter

        dest = Path(self.run_dir) / f"config{self.config_file_type}"
        adapter = get_adapter(str(dest))
        adapter.save(config, str(dest))
