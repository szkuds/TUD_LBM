"""Abstract base class for configuration file adapters.

Each adapter reads a specific file format (TOML, YAML, JSON, …) and
returns a :class:`SimulationConfig`.

Use :func:`get_adapter` to obtain the right adapter for a given file path::

    from config.adapter_base import get_adapter

    adapter = get_adapter("config.toml")
    config  = adapter.load("config.toml")
"""

from __future__ import annotations
import importlib
import os
from abc import ABC
from abc import abstractmethod
from typing import Any
import operators.force  # noqa: F401
from config.simulation_config import SimulationConfig
from registry import get_operator_names


class ConfigAdapter(ABC):
    """Abstract adapter that converts a config file into a SimulationConfig."""

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
            entry = dict(entry)  # shallow copy
            force_type = entry.get("type")

            if force_type is None:
                raise KeyError("Each [[force]] table must have a 'type' key.")

            if force_type not in get_operator_names("force"):
                registered = ", ".join(sorted(get_operator_names("force")))
                raise KeyError(
                    f"Unknown force type '{force_type}'. Registered types: {registered}",
                )

            validated.append(entry)
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
    _, ext = os.path.splitext(path)
    ext = ext.lower()

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
