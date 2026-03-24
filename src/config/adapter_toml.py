"""TOML configuration file adapter.

Reads and writes ``.toml`` config files.

Requires Python ≥ 3.11 (``tomllib`` in stdlib) **or** the ``tomli``
back-port on Python 3.10::

    pip install tomli

Example usage::

    from config.adapter_toml import TomlAdapter

    adapter = TomlAdapter()
    config  = adapter.load("example_for_test/config_simple.toml")
    adapter.save(config, "output/config.toml")
"""

from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Any
import tomli_w
import tomllib
from config.adapter_base import ConfigAdapter
from config.simulation_config import SimulationConfig


class TomlAdapter(ConfigAdapter):
    """Adapter that reads and writes TOML configuration files.

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

    @staticmethod
    def _apply_output_overrides(
        sim_table: dict[str, Any],
        output_table: dict[str, Any],
    ) -> None:
        """Merge ``[output]`` overrides into *sim_table* in-place."""
        for key, value in output_table.items():
            if key == "results_dir":
                value = str(Path(value).expanduser())
            sim_table[key] = value

    def load(self, path: str) -> SimulationConfig:
        """Parse *path* and return a :class:`SimulationConfig`.

        Args:
            path: Filesystem path to a ``.toml`` file.

        Returns:
            A validated :class:`SimulationConfig`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If required sections/keys are missing or invalid.
            KeyError: If a ``[[force]]`` type is not in the force registry.
        """
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("rb") as fh:
            raw = tomllib.load(fh)

        # ── [simulation_type] (required) ──────────────────────────────────
        sim_table = dict(raw.get("simulation_type", {}))
        if not sim_table:
            raise ValueError(
                f"Config file '{path}' is missing the required [simulation_type] table.",
            )

        sim_type: str = sim_table.pop("type", "single_phase")

        # ── Convert grid_shape from TOML array to tuple ──────────────
        if "grid_shape" in sim_table:
            sim_table["grid_shape"] = tuple(sim_table["grid_shape"])

        # ── Merge [multiphase] table ─────────────────────────────────
        if sim_type == "multiphase":
            multiphase_table = raw.get("multiphase", {})

            sim_table.update(multiphase_table)
        elif sim_type != "single_phase":
            raise ValueError(
                f"Unknown simulation type '{sim_type}'. Expected 'single_phase' or 'multiphase'.",
            )

        # ── [boundary_conditions] (optional) ─────────────────────────
        bc_config = raw.get("boundary_conditions")
        if bc_config is not None:
            sim_table["bc_config"] = dict(bc_config)

        # ── [[force]] (optional) ─────────────────────────────────────
        force_tables: list[dict[str, Any]] = raw.get("force", [])
        if force_tables:
            sim_table["force_enabled"] = True
            # Store as plain dicts — actual JAX force objects are built
            # later in ``build_setup()``.
            sim_table["force_config"] = self.parse_force_tables(force_tables)

        # ── [output] overrides ───────────────────────────────────────
        self._apply_output_overrides(sim_table, raw.get("output", {}))

        # ── Build SimulationConfig ───────────────────────────────────
        sim_table["sim_type"] = sim_type

        # Separate known fields from extra
        known_fields = {f.name for f in dataclasses.fields(SimulationConfig)}
        config_kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in sim_table.items():
            if k in known_fields:
                config_kwargs[k] = v
            else:
                extra[k] = v
        config_kwargs["extra"] = extra

        return SimulationConfig(**config_kwargs)

    def save(self, config: SimulationConfig, path: str) -> None:
        """Serialise *config* to a ``.toml`` file at *path*.

        Delegates the field → section bucketing to
        :meth:`~ConfigAdapter.build_sections` (shared by all adapters)
        and writes the result with ``tomli_w``.

        Args:
            config: A validated :class:`SimulationConfig`.
            path:   Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        doc = self.build_sections(config)

        with path.open("wb") as fh:
            tomli_w.dump(doc, fh)
