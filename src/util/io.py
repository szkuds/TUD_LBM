import logging
import sys
from dataclasses import fields as dc_fields
from datetime import datetime
from datetime import timezone
from pathlib import Path
from types import MethodType
from .output_data import output_writers


def _config_from_dict(d: dict) -> "SimulationConfig":
    """Build a :class:`SimulationConfig` from a ``to_dict()``-style dict.

    ``SimulationConfig.to_dict()`` adds ``simulation_type`` and merges
    ``extra`` into the top-level dict — both of which are not valid
    constructor kwargs.  This helper strips/remaps them before
    instantiation.
    """
    from config.simulation_config import SimulationConfig

    d = dict(d)  # shallow copy
    d.pop("simulation_type", None)  # added by to_dict(), not a ctor param

    # Normalise grid_shape to tuple
    if "grid_shape" in d and not isinstance(d["grid_shape"], tuple):
        d["grid_shape"] = tuple(d["grid_shape"])

    # Separate known fields from extras
    known = {f.name for f in dc_fields(SimulationConfig)}
    kwargs = {}
    extra = {}
    for k, v in d.items():
        if k in known:
            kwargs[k] = v
        else:
            extra[k] = v
    if extra:
        kwargs["extra"] = extra

    return SimulationConfig(**kwargs)


class SimulationIO:
    """Handles all I/O operations for the simulation, including logging and saving results."""

    def __init__(
        self,
        base_dir: str = "results",
        config: dict | None = None,
        simulation_name: str | None = None,
        output_format: str = "numpy",
        config_file_type: str = ".toml",
    ):
        """Initialises the IO handler.

        Args:
            base_dir (str): The base directory to store simulation results.
            config (Dict, optional): A dictionary containing the simulation configuration to save.
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
        """Save the simulation configuration to the run directory.

        Uses :func:`~config.adapter_base.get_adapter` to dispatch to the
        correct adapter based on :attr:`config_file_type`.
        """
        from config.adapter_base import get_adapter

        dest = Path(self.run_dir) / f"config{self.config_file_type}"
        adapter = get_adapter(str(dest))
        cfg = _config_from_dict(config)
        adapter.save(cfg, str(dest))

        print(f"Configuration saved to {dest}")
