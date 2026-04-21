"""TOML configuration file adapter.

Reads and writes ``.toml`` config files.

Requires Python ≥ 3.11 (``tomllib`` in stdlib) **or** the ``tomli``
back-port on Python 3.10::

    pip install tomli

Example usage::

    from config.adapter_toml import TomlAdapter

    adapter = TomlAdapter()
    config = adapter.load("example_for_test/config_simple.toml")
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
        Required. Contains the simulation type (``type``), grid shape,
        physics parameters, and runner/IO fields.

    ``[multiphase]``
        Optional. Extra physics parameters when ``type = "multiphase"``.

    ``[gravity_force]`` / ``[electric_force]`` / ...
        Optional. One top-level table per force model.

    ``[boundary_conditions]``
        Optional. Boundary condition configuration (including nested
        ``wetting_params`` and ``hysteresis_params``).

    ``[output]``
        Optional. Output/saving overrides (``results_dir``, ``fields``).
    """

    @staticmethod
    def _apply_output_overrides(
        sim_table: dict[str, Any],
        output_table: dict[str, Any],
    ) -> None:
        """Merge ``[output]`` overrides into *sim_table* in-place."""
        for key, value in output_table.items():
            if key == "results_dir":
                sim_table[key] = str(Path(value).expanduser())
            else:
                sim_table[key] = value

    @staticmethod
    def _validate_and_process_forces(
        raw: dict[str, Any],
        sim_table: dict[str, Any],
    ) -> None:
        """Extract and validate force sections from raw TOML.

        Args:
            raw: Raw TOML dictionary.
            sim_table: Mutable simulation table to update in-place.

        Raises:
            KeyError: If force type is unknown.
            TypeError: If force section is not a table.
        """
        known_force_fields = {f.name for f in dataclasses.fields(SimulationConfig) if f.name.endswith("_force")}
        for key, value in raw.items():
            if not key.endswith("_force"):
                continue
            if key not in known_force_fields:
                raise KeyError(f"Unknown force type '{key}'")
            if not isinstance(value, dict):
                raise TypeError(f"Force section '[{key}]' must be a table, got {type(value).__name__}.")
            sim_table[key] = dict(value)

    def load_raw(self, path: str) -> dict[str, Any]:
        """Parse *path* and return raw configuration dict.

        Unlike :meth:`load`, this returns the raw config dict without
        instantiating :class:`SimulationConfig`.  Useful for detecting
        array fields before expansion.

        Args:
            path: Filesystem path to a ``.toml`` file.

        Returns:
            Raw config dictionary (may contain array values for physics
            parameters, plus output overrides from ``[output]``).

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If required sections/keys are missing.
        """
        path_obj: Path = Path(path).expanduser()
        if not path_obj.is_file():
            raise FileNotFoundError(f"Config file not found: {path_obj}")

        with path_obj.open("rb") as fh:
            raw = tomllib.load(fh)

        # ── [simulation_type] (required) ──────────────────────────────────
        sim_table = dict(raw.get("simulation_type", {}))
        if not sim_table:
            raise ValueError(f"Config file '{path}' is missing the required [simulation_type] table.")

        sim_type: str = sim_table.pop("type", "single_phase")

        # ── Convert grid_shape from TOML array to tuple only if scalar ─
        if "grid_shape" in sim_table and not isinstance(sim_table["grid_shape"], list):
            sim_table["grid_shape"] = tuple(sim_table["grid_shape"])

        # ── Validate sim_type ────────────────────────────────────────
        valid_types = ("single_phase", "multiphase")
        if sim_type not in valid_types:
            raise ValueError(f"Unknown simulation type '{sim_type}'. Expected one of: {', '.join(valid_types)}.")

        # ── Merge [multiphase] table (for multiphase simulations) ────
        if sim_type == "multiphase":
            multiphase_table = raw.get("multiphase", {})
            sim_table.update(multiphase_table)

        # ── [boundary_conditions] (optional) ─────────────────────────
        bc_config = raw.get("boundary_conditions")
        if bc_config is not None:
            sim_table["bc_config"] = dict(bc_config)

        # ── Wetting sections ─────────────────────────────────────────
        if "wetting" in raw:
            sim_table["wetting_config"] = dict(raw["wetting"])
        if "hysteresis" in raw:
            sim_table["hysteresis_config"] = dict(raw["hysteresis"])

        if "initialisation" in raw:
            sim_table["initialisation"] = dict(raw["initialisation"])

        # ── Force sections ──────────────────────────────────────────
        self._validate_and_process_forces(raw, sim_table)

        # ── Add output overrides (from [output] section) ──────────────
        self._apply_output_overrides(sim_table, raw.get("output", {}))

        # ── Build sim_type field ────────────────────────────────────
        sim_table["sim_type"] = sim_type

        return sim_table

    def load(self, path: str) -> SimulationConfig:
        """Parse *path* and return a :class:`SimulationConfig`.

        Supports both scalar and array-valued parameters. For array
        parameters, use :meth:`load_raw` followed by
        :func:`~config.array_expansion.expand_config` to generate
        multiple configs.

        Args:
            path: Filesystem path to a ``.toml`` file.

        Returns:
            A validated :class:`SimulationConfig`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If required sections/keys are missing or invalid.
        """
        sim_table = self.load_raw(path)

        # ── Build SimulationConfig ───────────────────────────────────

        # Separate known fields from extra
        known_fields = {f.name for f in dataclasses.fields(SimulationConfig)}
        config_kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = dict(sim_table.get("extra", {}))
        for k, v in sim_table.items():
            if k == "extra":
                continue
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
            path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
        path_obj: Path = Path(path).expanduser()
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        doc = self.build_sections(config)

        with path_obj.open("wb") as fh:
            tomli_w.dump(doc, fh)
