"""Plotting utilities for TUD-LBM.

Public surface
--------------
FigureBuilder  — assembles per-timestep composite figures from config.
PlotOperator   — abstract base class for individual panel operators.
visualise      — backward-compatible entry point for simulation objects
                 or run directories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Trigger operator self-registration at import time.
from . import analysis as _analysis_mod  # noqa: F401
from . import density as _density_mod  # noqa: F401
from . import force as _force_mod  # noqa: F401
from . import velocity as _velocity_mod  # noqa: F401
from .base import PlotOperator
from .figure_builder import FigureBuilder


def _load_legacy_config(run_dir: Path) -> dict[str, Any]:
    """Load plotting config from a run directory using legacy filenames."""
    toml_path = run_dir / "app_setup.toml"
    json_candidates = [run_dir / "app_setup.json", run_dir / "config.json"]

    if toml_path.exists():
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[no-redef]

        with open(toml_path, "rb") as fh:
            raw = tomllib.load(fh)

        config: dict[str, Any] = dict(raw.get("simulation_type", {}))
        config["simulation_type"] = config.get("type", "single_phase")
        config.pop("type", None)
        config.update(raw.get("multiphase", {}))
        config.update(raw.get("output", {}))
        if "fields" in config and "save_fields" not in config:
            config["save_fields"] = list(config.pop("fields"))
        if "plots" in config and "plot_fields" not in config:
            config["plot_fields"] = list(config.pop("plots"))
        if "boundary_conditions" in raw:
            config["boundary_conditions"] = raw["boundary_conditions"]
        return config

    for cfg_path in json_candidates:
        if cfg_path.exists():
            with open(cfg_path) as fh:
                return json.load(fh)

    return {}


def visualise(sim_instance_or_run_dir, title: str = "LBM Simulation Results", skip: int = 0) -> None:
    """Backward-compatible plotting entry point.

    Accepts either the historical simulation instance object or a run-directory
    path. In both cases, figures are rendered by :class:`FigureBuilder`.
    """
    if hasattr(sim_instance_or_run_dir, "io_handler"):
        sim_instance = sim_instance_or_run_dir
        run_dir = Path(sim_instance.io_handler.run_dir)
        config = dict(getattr(sim_instance, "config", {}) or {})
        if title and "plot_title" not in config:
            config["plot_title"] = title
    else:
        run_dir = Path(sim_instance_or_run_dir)
        config = _load_legacy_config(run_dir)
        if title and "plot_title" not in config:
            config["plot_title"] = title

    builder = FigureBuilder(config=config, run_dir=run_dir)
    builder.build_all(skip=skip)


__all__ = ["FigureBuilder", "PlotOperator", "visualise"]
