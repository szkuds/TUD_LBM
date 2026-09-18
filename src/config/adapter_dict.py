"""Dict configuration adapter.

Accepts a plain Python ``dict`` and returns a
:class:`~config.simulation_config.SimulationConfig`.

Usage::

    from config.adapter_dict import DictAdapter

    adapter = DictAdapter()
    config  = adapter.load({"grid_shape": [64, 64], "tau": 0.8})
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from src.config.adapter_base import ConfigAdapter

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig


class DictAdapter(ConfigAdapter):
    """Adapter that builds a :class:`SimulationConfig` from a dict.

    Accepts two shapes:

    **Flat** (SimulationConfig field names at top level)::

        {"sim_type": "multiphase", "kappa": 0.04, ...}

    **Nested** (mirrors TOML section structure)::

        {"simulation_type": {"type": "multiphase", ...}, "multiphase": {...}, ...}
    """

    def load_raw(self, source: dict[str, Any]) -> dict[str, Any]:
        """Return a flat config dict from *source*, handling both flat and nested shapes."""
        if isinstance(source.get("simulation_type"), dict):
            return self._merge_sections(source)
        return dict(source)

    def save(self, config: SimulationConfig, path: str) -> None:
        """Save a simulation config to *path*."""
        del config, path
        msg = "DictAdapter does not write files; use TomlAdapter."
        raise NotImplementedError(msg)
