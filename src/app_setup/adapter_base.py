"""Abstract base class for configuration file adapters.

Each adapter reads a specific file format (TOML, YAML, JSON, …) and
returns a :class:`SimulationSetup`.

Use :func:`get_adapter` to obtain the right adapter for a given file path::

    from app_setup.adapter_base import get_adapter

    adapter = get_adapter("app_setup.toml")
    setup   = adapter.load("app_setup.toml")
"""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from app_setup.saving_config import FORCE_REGISTRY
from app_setup.simulation_setup import SimulationSetup


class ConfigAdapter(ABC):
    """Abstract adapter that converts a config file into a SimulationSetup."""

    @abstractmethod
    def load(self, path: str) -> SimulationSetup:
        """Read *path* and return a fully-validated :class:`SimulationSetup`.

        Args:
            path: Filesystem path to the configuration file.

        Returns:
            A :class:`SimulationSetup` ready to be passed to :class:`Run`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file contents are invalid.
        """

    # ------------------------------------------------------------------
    # Shared helpers available to all adapters
    # ------------------------------------------------------------------

    @staticmethod
    def instantiate_forces(
        force_tables: List[Dict[str, Any]],
        grid_shape: Tuple[int, ...],
    ) -> List[Any]:
        """Create force objects from a list of dicts (e.g. ``[[force]]`` tables).

        Each dict **must** contain a ``type`` key whose value is looked up
        in :data:`~app_setup.saving_config.FORCE_REGISTRY`.  The remaining
        keys are forwarded to the force constructor as ``**kwargs``.

        ``grid_shape`` is always injected so TOML files don't need to
        repeat it.

        Args:
            force_tables: List of dicts, each with at least a ``type`` key.
            grid_shape:   Grid dimensions, injected into every constructor.

        Returns:
            A list of instantiated force objects.

        Raises:
            KeyError: If a ``type`` is not found in the registry.
        """
        forces: List[Any] = []
        for entry in force_tables:
            entry = dict(entry)  # shallow copy so we don't mutate the caller
            force_type = entry.pop("type")

            if force_type not in FORCE_REGISTRY:
                registered = ", ".join(sorted(FORCE_REGISTRY))
                raise KeyError(
                    f"Unknown force type '{force_type}'. "
                    f"Registered types: {registered}"
                )

            module_path, class_name = FORCE_REGISTRY[force_type]
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)

            # Inject grid_shape (every force needs it)
            entry.setdefault("grid_shape", grid_shape)

            forces.append(cls(**entry))
        return forces


# ------------------------------------------------------------------
# Public factory
# ------------------------------------------------------------------

_ADAPTER_MAP: Dict[str, str] = {
    ".toml": "app_setup.adapter_toml.TomlAdapter",
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
            f"Unsupported app_setup file extension '{ext}'. "
            f"Supported: {supported}"
        )

    fqn = _ADAPTER_MAP[ext]
    module_path, class_name = fqn.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()

