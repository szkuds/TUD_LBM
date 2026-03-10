"""TOML configuration file adapter.

Reads a ``.toml`` file and returns a :class:`SimulationBundle`.

Requires Python ≥ 3.11 (``tomllib`` in stdlib) **or** the ``tomli``
back-port on Python 3.10::

    pip install tomli

Example usage::

    from app_setup.adapter_toml import TomlAdapter

    adapter = TomlAdapter()
    bundle  = adapter.load("example/config_simple.toml")
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, List, Tuple

try:
    import tomllib  # Python ≥ 3.11
except ModuleNotFoundError:  # pragma: no cover — Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from app_setup.adapter_base import ConfigAdapter
from app_setup.simulation_config import (
    MultiphaseConfig,
    RunnerConfig,
    SimulationBundle,
    SinglePhaseConfig,
)


class TomlAdapter(ConfigAdapter):
    """Adapter that reads TOML configuration files.

    Supported top-level tables
    --------------------------
    ``[simulation_type]``
        Required.  Contains the simulation_type type (``type``), grid shape,
        physics parameters, and runner/IO fields that are split into
        :class:`RunnerConfig`.

    ``[multiphase]``
        Optional.  Extra physics parameters when ``type = "multiphase"``.

    ``[[force]]``
        Optional.  One or more force definitions (array-of-tables).

    ``[boundary_conditions]``
        Optional.  Boundary condition configuration (including nested
        ``wetting_params`` and ``hysteresis_params``).

    ``[output]``
        Optional.  Output/saving overrides (``results_dir``, ``fields``).
    """

    # Keys in [simulation_type] that belong to RunnerConfig, not the physics app_setup
    _RUNNER_KEYS = frozenset({
        "save_interval",
        "skip_interval",
        "init_type",
        "init_dir",
        "simulation_name",
        "save_fields",
    })

    # Keys in [simulation_type] that are adapter meta-data, not forwarded
    _META_KEYS = frozenset({
        "type",
    })

    def load(self, path: str) -> SimulationBundle:
        """Parse *path* and return a :class:`SimulationBundle`.

        Args:
            path: Filesystem path to a ``.toml`` file.

        Returns:
            A validated :class:`SimulationBundle`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If required sections/keys are missing or invalid.
            KeyError: If a ``[[force]]`` type is not in the force registry.
        """
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as fh:
            raw = tomllib.load(fh)

        # ── [simulation_type] (required) ──────────────────────────────────
        sim_table = dict(raw.get("simulation_type", {}))
        if not sim_table:
            raise ValueError(
                f"Config file '{path}' is missing the required [simulation_type] table."
            )

        sim_type: str = sim_table.pop("type", "single_phase")

        # ── Split runner keys out of [simulation_type] ────────────────────
        runner_kwargs = self._extract_runner_kwargs(sim_table)

        # ── [output] overrides ───────────────────────────────────────
        output_table = raw.get("output", {})
        if "results_dir" in output_table:
            runner_kwargs["results_dir"] = os.path.expanduser(
                output_table["results_dir"]
            )
        if "fields" in output_table:
            runner_kwargs["save_fields"] = list(output_table["fields"])

        runner = RunnerConfig(**runner_kwargs)

        # ── Build physics app_setup ─────────────────────────────────────
        if sim_type == "single_phase":
            simulation = self._build_single_phase(sim_table, raw)
        elif sim_type == "multiphase":
            simulation = self._build_multiphase(sim_table, raw)
        else:
            raise ValueError(
                f"Unknown simulation_type type '{sim_type}'. "
                f"Expected 'single_phase' or 'multiphase'."
            )

        return SimulationBundle(simulation=simulation, runner=runner)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_runner_kwargs(
        self, sim_table: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Pop runner-related keys from *sim_table* and return them as a dict."""
        kwargs: Dict[str, Any] = {}
        for key in list(sim_table):
            if key in self._RUNNER_KEYS:
                kwargs[key] = sim_table.pop(key)
        return kwargs

    def _ensure_grid_tuple(
        self, sim_table: Dict[str, Any]
    ) -> None:
        """Convert ``grid_shape`` from a TOML array (list) to a tuple in place."""
        if "grid_shape" in sim_table:
            sim_table["grid_shape"] = tuple(sim_table["grid_shape"])

    def _build_single_phase(
        self,
        sim_table: Dict[str, Any],
        raw: Dict[str, Any],
    ) -> SinglePhaseConfig:
        """Construct a :class:`SinglePhaseConfig` from the parsed TOML data."""
        self._ensure_grid_tuple(sim_table)

        # Boundary conditions (optional)
        bc_config = raw.get("boundary_conditions")
        if bc_config is not None:
            sim_table["bc_config"] = dict(bc_config)

        # Forces (optional)
        force_tables: List[Dict[str, Any]] = raw.get("force", [])
        if force_tables:
            grid_shape: Tuple[int, ...] = sim_table["grid_shape"]
            sim_table["force_enabled"] = True
            sim_table["force_obj"] = self.instantiate_forces(
                force_tables, grid_shape
            )

        # Separate known fields from extra
        known_fields = {f.name for f in dataclasses.fields(SinglePhaseConfig)}
        config_kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in sim_table.items():
            if k in known_fields:
                config_kwargs[k] = v
            else:
                extra[k] = v
        config_kwargs["extra"] = extra

        return SinglePhaseConfig(**config_kwargs)

    def _build_multiphase(
        self,
        sim_table: Dict[str, Any],
        raw: Dict[str, Any],
    ) -> MultiphaseConfig:
        """Construct a :class:`MultiphaseConfig` from the parsed TOML data."""
        self._ensure_grid_tuple(sim_table)

        # Merge [multiphase] table into sim_table
        multiphase_table = raw.get("multiphase", {})
        sim_table.update(multiphase_table)

        # Boundary conditions (optional)
        bc_config = raw.get("boundary_conditions")
        if bc_config is not None:
            sim_table["bc_config"] = dict(bc_config)

        # Forces (optional)
        force_tables: List[Dict[str, Any]] = raw.get("force", [])
        if force_tables:
            grid_shape: Tuple[int, ...] = sim_table["grid_shape"]
            sim_table["force_enabled"] = True
            sim_table["force_obj"] = self.instantiate_forces(
                force_tables, grid_shape
            )

        # Separate known fields from extra
        known_fields = {f.name for f in dataclasses.fields(MultiphaseConfig)}
        config_kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in sim_table.items():
            if k in known_fields:
                config_kwargs[k] = v
            else:
                extra[k] = v
        config_kwargs["extra"] = extra

        return MultiphaseConfig(**config_kwargs)

