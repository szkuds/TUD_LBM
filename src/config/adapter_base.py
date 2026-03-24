"""Abstract base class for configuration file adapters.

Each adapter reads a specific file format (TOML, YAML, JSON, …) and
returns a :class:`SimulationConfig`.

Use :func:`get_adapter` to obtain the right adapter for a given file path::

    from config.adapter_base import get_adapter

    adapter = get_adapter("config.toml")
    config  = adapter.load("config.toml")
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
    :meth:`save` (config → file).  The shared :meth:`build_sections`
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
            path:   Destination file path (extension must match the adapter).

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
        return {
            f.name: f.metadata.get(CONFIG_SECTION, "simulation_type")
            for f in dataclasses.fields(SimulationConfig)
        }

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
    def build_sections(cls, config: SimulationConfig) -> dict[str, Any]:
        """Build a format-agnostic nested dict from *config*.

        The returned dict mirrors the canonical config-file layout::

            {
                "simulation_type": { "type": "single_phase", ... },
                "multiphase":      { ... },            # if applicable
                "boundary_conditions": { ... },        # if present
                "force":           [ {...}, ... ],      # if present
                "output":          { ... },             # if present
            }

        Each adapter's :meth:`save` can pass this dict straight to its
        format-specific serialiser (``tomli_w.dump``, ``yaml.dump``,
        ``json.dump``, …).

        Args:
            config: A validated :class:`SimulationConfig`.

        Returns:
            A nested mapping suitable for serialisation.
        """
        d = dataclasses.asdict(config)
        sections = cls._section_map()
        sim_type = d.get("sim_type", "single_phase")

        # ── Bucket every field by its declared section ────────────────
        sim_table: dict[str, Any] = {"type": sim_type}
        multiphase_table: dict[str, Any] = {}
        output_table: dict[str, Any] = {}
        bc_out: dict[str, Any] = {}
        force_list: list[dict[str, Any]] = []

        for key, value in d.items():
            section = sections.get(key, "simulation_type")

            # identity / extra are handled specially, not written as-is
            if section in {"identity", "extra"}:
                continue

            if value is None:
                continue

            if section == "multiphase" and sim_type == "multiphase":
                multiphase_table[key] = value
            elif section == "output":
                output_table[key] = value
            elif section == "boundary_conditions":
                if key == "bc_config":
                    bc_out.update(value)
                elif key == "wetting_config":
                    bc_out["wetting_params"] = value
                elif key == "hysteresis_config":
                    bc_out["hysteresis_params"] = value
                else:
                    bc_out[key] = value
            elif section == "force":
                if key == "force_config":
                    force_list = list(value) if isinstance(value, list) else [value]
                # force_enabled is derived — skip
            elif section == "simulation_type":
                sim_table[key] = cls._serialize_safe(value)

        # Merge extra keys into the simulation_type table
        for key, value in d.get("extra", {}).items():
            sim_table[key] = cls._serialize_safe(value)

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
        :data:`VALID_FORCE_TYPES`.  The dicts are stored as-is in
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
                raise KeyError(
                    f"Unknown force type '{force_type}'. Registered types: {registered}",
                )

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
        raise ValueError(
            f"Unsupported config file extension '{ext}'. Supported: {supported}",
        )

    fqn = _ADAPTER_MAP[ext]
    module_path, class_name = fqn.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()
