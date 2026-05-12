"""Base class for configuration file adapters."""

from __future__ import annotations
import dataclasses
import importlib
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any
from tud_lbm.config.simulation_config import CONFIG_SECTION
from tud_lbm.config.simulation_config import SimulationConfig


class ConfigAdapter(ABC):
    """Converts a config file into a SimulationConfig and back."""

    @abstractmethod
    def load_raw(self, path: str) -> dict[str, Any]:
        """Parse *path* and return raw configuration dict.

        Unlike :meth:`load`, this returns the raw config dict without
        instantiating :class:`SimulationConfig`. Useful for detecting
        array fields before expansion.

        Args:
            path: Filesystem path to a configuration file.

        Returns:
            Raw config dictionary.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> SimulationConfig:
        """Parse *path* and return a :class:`SimulationConfig`.

        Args:
            path: Filesystem path to a configuration file.

        Returns:
            A validated :class:`SimulationConfig`.
        """
        ...

    @abstractmethod
    def save(self, config: SimulationConfig, path: str) -> None:
        """Save a :class:`SimulationConfig` to *path*.

        Args:
            config: Configuration to save.
            path: Filesystem path where to save the configuration.
        """
        ...

    @staticmethod
    def _serialize_safe(value: Any) -> Any:  # noqa: ANN401
        """Convert tuples to lists and recursively process nested structures."""
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, dict):
            return {k: ConfigAdapter._serialize_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ConfigAdapter._serialize_safe(v) for v in value]
        return value

    @classmethod
    def build_sections(cls, config: SimulationConfig) -> dict[str, Any]:
        """Build a format-agnostic nested dict from *config*, routed by CONFIG_SECTION metadata."""
        sections = {
            f.name: f.metadata.get(CONFIG_SECTION, "simulation_type") for f in dataclasses.fields(SimulationConfig)
        }
        sim_type = config.sim_type
        skip = {"identity", "extra"}

        buckets: dict[str, dict[str, Any]] = defaultdict(dict)
        for key, value in dataclasses.asdict(config).items():
            section = sections.get(key, "simulation_type")
            if value is None or section in skip:
                continue
            if section == "multiphase" and sim_type != "multiphase":
                continue
            if isinstance(value, dict):
                buckets[section].update(cls._serialize_safe(value))
            else:
                buckets[section][key] = cls._serialize_safe(value)

        buckets["simulation_type"]["type"] = sim_type
        for ek, ev in (config.extra or {}).items():
            buckets["simulation_type"][ek] = cls._serialize_safe(ev)

        return {
            "simulation_type": buckets.pop("simulation_type", {}),
            **{k: buckets[k] for k in sorted(buckets) if buckets[k]},
        }


_ADAPTER_MAP: dict[str, str] = {
    ".toml": "tud_lbm.config.adapter_toml.TomlAdapter",
}


def get_adapter(path: str) -> ConfigAdapter:
    """Return the appropriate adapter for *path* based on file extension."""
    ext = Path(path).suffix.lower()
    fqn = _ADAPTER_MAP.get(ext)
    if not fqn:
        msg = f"Unsupported extension '{ext}'. Supported: {', '.join(sorted(_ADAPTER_MAP))}"
        raise ValueError(msg)
    module_path, class_name = fqn.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)()
