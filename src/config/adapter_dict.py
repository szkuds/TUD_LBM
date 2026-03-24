"""Dict configuration adapter.

Accepts a plain Python ``dict`` and returns a
:class:`~config.simulation_config.SimulationConfig`.

Usage::

    from config.adapter_dict import DictAdapter

    adapter = DictAdapter()
    config  = adapter.load({"grid_shape": [64, 64], "tau": 0.8})
"""

from __future__ import annotations
from typing import Any
from config.simulation_config import SimulationConfig


class DictAdapter:
    """Adapter that builds a :class:`SimulationConfig` from a dict."""

    def load(self, d: dict[str, Any]) -> SimulationConfig:
        """Build a :class:`SimulationConfig` from *d*.

        Normalises common key formats (``grid_shape`` list → tuple)
        and passes everything through to the ``SimulationConfig``
        constructor, which performs full validation.

        Args:
            d: Configuration dict.

        Returns:
            A validated :class:`SimulationConfig`.
        """
        d = dict(d)  # shallow copy

        # Normalise grid_shape to tuple
        if "grid_shape" in d and not isinstance(d["grid_shape"], tuple):
            d["grid_shape"] = tuple(d["grid_shape"])

        return SimulationConfig(**d)
