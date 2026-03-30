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
from collections import defaultdict
from pathlib import Path
from typing import Any
from config.simulation_config import CONFIG_SECTION
from config.simulation_config import SimulationConfig


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
    def build_sections(cls, config: SimulationConfig) -> dict[str, Any]:
        """Build a format-agnostic nested dict from *config*.

        Section routing is driven entirely by CONFIG_SECTION metadata.
        No per-section handler methods needed — every field drops into
        its declared section bucket.
        """
        d = dataclasses.asdict(config)
        sections = cls._section_map()
        sim_type = d.get("sim_type", "single_phase")

        # Accumulator: one dict per section name
        buckets: dict[str, dict[str, Any]] = defaultdict(dict)

        for key, value in d.items():
            if value is None:
                continue
            section = sections.get(key, "simulation_type")

            if section in {"identity", "extra"}:
                continue

            if section == "multiphase" and sim_type != "multiphase":
                continue

            # Everything — including forces — drops into its section bucket
            buckets[section][key] = cls._serialize_safe(value)

        buckets["simulation_type"]["type"] = sim_type

        for ek, ev in d.get("extra", {}).items():
            buckets["simulation_type"][ek] = cls._serialize_safe(ev)

        doc: dict[str, Any] = {"simulation_type": buckets.pop("simulation_type", {})}
        for section_name in sorted(buckets):
            if buckets[section_name]:
                doc[section_name] = buckets[section_name]

        return doc


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
