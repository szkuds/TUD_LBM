"""TOML configuration file adapter.

Reads a ``.toml`` file and returns a :class:`SimulationSetup`.

Requires Python ≥ 3.11 (``tomllib`` in stdlib) **or** the ``tomli``
back-port on Python 3.10::

    pip install tomli

Example usage::

    from app_setup.adapter_toml import TomlAdapter

    adapter = TomlAdapter()
    setup   = adapter.load("example/config_simple.toml")
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
from app_setup.simulation_setup import SimulationSetup


class TomlAdapter(ConfigAdapter):
    """Adapter that reads TOML configuration files.

    Supported top-level tables
    --------------------------
    ``[simulation_type]``
        Required.  Contains the simulation type (``type``), grid shape,
        physics parameters, and runner/IO fields.

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

    # Keys in [simulation_type] that are adapter meta-data, not forwarded
    _META_KEYS = frozenset({
        "type",
    })

    def load(self, path: str) -> SimulationSetup:
        """Parse *path* and return a :class:`SimulationSetup`.

        Args:
            path: Filesystem path to a ``.toml`` file.

        Returns:
            A validated :class:`SimulationSetup`.

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

        # ── Convert grid_shape from TOML array to tuple ──────────────
        if "grid_shape" in sim_table:
            sim_table["grid_shape"] = tuple(sim_table["grid_shape"])

        # ── Merge [multiphase] table ─────────────────────────────────
        if sim_type == "multiphase":
            multiphase_table = raw.get("multiphase", {})

            # Explicitly reject legacy `width` key to avoid silent misuse.
            if "width" in multiphase_table:
                raise KeyError(
                    "Legacy key 'width' is no longer supported in the "
                    "[multiphase] table; please use 'interface_width' instead."
                )

            sim_table.update(multiphase_table)
        elif sim_type not in ("single_phase",):
            raise ValueError(
                f"Unknown simulation type '{sim_type}'. "
                f"Expected 'single_phase' or 'multiphase'."
            )

        # ── [boundary_conditions] (optional) ─────────────────────────
        bc_config = raw.get("boundary_conditions")
        if bc_config is not None:
            sim_table["bc_config"] = dict(bc_config)

        # ── [[force]] (optional) ─────────────────────────────────────
        force_tables: List[Dict[str, Any]] = raw.get("force", [])
        if force_tables:
            grid_shape: Tuple[int, ...] = sim_table["grid_shape"]
            sim_table["force_enabled"] = True
            sim_table["force_obj"] = self.instantiate_forces(
                force_tables, grid_shape
            )

        # ── [output] overrides ───────────────────────────────────────
        output_table = raw.get("output", {})
        if "results_dir" in output_table:
            sim_table["results_dir"] = os.path.expanduser(
                output_table["results_dir"]
            )
        if "fields" in output_table:
            sim_table["save_fields"] = list(output_table["fields"])

        # ── Build SimulationSetup ────────────────────────────────────
        sim_table["sim_type"] = sim_type

        # Separate known fields from extra
        known_fields = {f.name for f in dataclasses.fields(SimulationSetup)}
        config_kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in sim_table.items():
            if k in known_fields:
                config_kwargs[k] = v
            else:
                extra[k] = v
        config_kwargs["extra"] = extra

        return SimulationSetup(**config_kwargs)
