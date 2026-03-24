"""Abstract base class for configuration file adapters.

Each adapter reads a specific file format (TOML, YAML, JSON, …) and
returns a :class:`SimulationConfig`.

Use :func:`get_adapter` to obtain the right adapter for a given file path::

    from config.adapter_base import get_adapter

    adapter = get_adapter("config.toml")
    config = adapter.load("config.toml")
"""

from __future__ import annotations
import dataclasses
import importlib
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
import operators.force  # noqa: F401
from config.simulation_config import CONFIG_SECTION
from config.simulation_config import SimulationConfig
from registry import get_operator_names


class ConfigAdapter(ABC):
    """Abstract adapter that converts a config file into a SimulationConfig.

    Concrete adapters must implement :meth:`load` (file → config) and
    :meth:`save` (config → file). The shared :meth:`build_sections`
    helper provides a format-agnostic structured dict that any
    ``save()`` implementation can serialise directly.
    """

    @abstractmethod
    def load(self, path: str) -> SimulationConfig:
        """Read *path* and return a fully-validated :class:`SimulationConfig`.

        Args:
            path: Filesystem path to the configuration file.

        Returns:
            A :class:`SimulationConfig` ready to be passed to ``build_setup``.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file contents are invalid.
        """

    @abstractmethod
    def save(self, config: SimulationConfig, path: str) -> None:
        """Serialise *config* and write it to *path*.

        Args:
            config: A validated :class:`SimulationConfig`.
            path: Destination file path (extension must match the adapter).

        Raises:
            OSError: If the file cannot be written.
        """

    # ------------------------------------------------------------------
    # Shared serialisation helpers (used by every adapter's save())
    # ------------------------------------------------------------------

    @staticmethod
    def _section_map() -> dict[str, str]:
        """Return ``{field_name: config_section}`` derived from field metadata.

        Fields without a ``config_section`` metadata entry default to
        ``"simulation_type"``.
        """
        return {f.name: f.metadata.get(CONFIG_SECTION, "simulation_type") for f in dataclasses.fields(SimulationConfig)}

    @staticmethod
    def _serialize_safe(value: Any) -> Any:
        """Convert Python values to serialisation-safe types.

        Tuples become lists (most formats lack a tuple type) and nested
        structures are recursively converted.
        """
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, dict):
            return {k: ConfigAdapter._serialize_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ConfigAdapter._serialize_safe(v) for v in value]
        return value

    @classmethod
    def _handle_boundary_conditions(
        cls,
        key: str,
        value: Any,
        bc_out: dict[str, Any],
    ) -> None:
        if key == "bc_config":
            bc_out.update(value)
        elif key == "wetting_config":
            bc_out["wetting_params"] = value
        elif key == "hysteresis_config":
            bc_out["hysteresis_params"] = value
        else:
            bc_out[key] = value

    @classmethod
    def _handle_force(
        cls,
        key: str,
        value: Any,
        force_list: list[dict[str, Any]],
    ) -> None:
        if key == "force_config":
            items = list(value) if isinstance(value, list) else [value]
            force_list[:] = items

    @classmethod
    def _handle_simulation_type(
        cls,
        key: str,
        value: Any,
        sim_table: dict[str, Any],
    ) -> None:
        sim_table[key] = cls._serialize_safe(value)

    @classmethod
    def build_sections(cls, config: SimulationConfig) -> dict[str, Any]:
        """Build a format-agnostic nested dict from *config*."""
        d = dataclasses.asdict(config)
        sections = cls._section_map()
        sim_type = d.get("sim_type", "single_phase")

        # ── Bucket every field by its declared section ────────────────
        sim_table: dict[str, Any] = {"type": sim_type}
        multiphase_table: dict[str, Any] = {}
        output_table: dict[str, Any] = {}
        bc_out: dict[str, Any] = {}
        force_list: list[dict[str, Any]] = []

        def _iter_relevant_items():
            for key, value in d.items():
                section = sections.get(key, "simulation_type")
                if section in {"identity", "extra"}:
                    continue
                if value is None:
                    continue
                yield key, value, section

        for key, value, section in _iter_relevant_items():
            if section == "multiphase" and sim_type == "multiphase":
                multiphase_table[key] = value
                continue

            if section == "output":
                output_table[key] = value
                continue

            if section == "boundary_conditions":
                cls._handle_boundary_conditions(key, value, bc_out)
                continue

            if section == "force":
                cls._handle_force(key, value, force_list)
                continue

            if section == "simulation_type":
                cls._handle_simulation_type(key, value, sim_table)

        # Merge extra keys into the simulation_type table
        for extra_key, extra_value in d.get("extra", {}).items():
            sim_table[extra_key] = cls._serialize_safe(extra_value)

        # ── Assemble top-level document ───────────────────────────────
        doc: dict[str, Any] = {"simulation_type": sim_table}

        if multiphase_table:
            doc["multiphase"] = multiphase_table
        if bc_out:
            doc["boundary_conditions"] = bc_out
        if force_list:
            doc["force"] = force_list
        if output_table:
            doc["output"] = output_table

        return doc

    # ------------------------------------------------------------------
    # Shared helpers available to all adapters
    # ------------------------------------------------------------------

    @staticmethod
    def parse_force_tables(
        force_tables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate and normalise ``[[force]]`` tables into plain dicts.

        Each dict **must** contain a ``type`` key whose value is one of
        :data:`VALID_FORCE_TYPES`. The dicts are stored as-is in
        ``SimulationConfig.force_config`` and later consumed by
        ``build_setup()`` to construct JAX-friendly force objects.

        Args:
            force_tables: List of dicts, each with at least a ``type`` key.

        Returns:
            A list of validated force configuration dicts.

        Raises:
            KeyError: If a ``type`` is not recognised.
        """
        validated: list[dict[str, Any]] = []
        for entry in force_tables:
            processed = dict(entry)  # shallow copy
            force_type = processed.get("type")

            if force_type is None:
                raise KeyError("Each [[force]] table must have a 'type' key.")

            if force_type not in get_operator_names("force"):
                registered = ", ".join(sorted(get_operator_names("force")))
                raise KeyError(f"Unknown force type '{force_type}'. Registered types: {registered}")

            validated.append(processed)
        return validated


# ------------------------------------------------------------------
# Public factory
# ------------------------------------------------------------------

_ADAPTER_MAP: dict[str, str] = {
    ".toml": "config.adapter_toml.TomlAdapter",
}


def get_adapter(path: str) -> ConfigAdapter:
    """Return the appropriate :class:`ConfigAdapter` for *path*.

    Dispatches on the file extension.

    Args:
        path: Filesystem path (only the suffix matters).

    Returns:
        An adapter instance.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = Path(path).suffix.lower()

    if ext not in _ADAPTER_MAP:
        supported = ", ".join(sorted(_ADAPTER_MAP))
        raise ValueError(f"Unsupported config file extension '{ext}'. Supported: {supported}")

    fqn = _ADAPTER_MAP[ext]
    module_path, class_name = fqn.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()
