"""Configuration package for TUD-LBM.

Public API::

    from config import SimulationConfig, from_toml, from_dict
    from config import TomlAdapter, DictAdapter
"""

from __future__ import annotations
from typing import Any
from typing import Dict
from config.adapter_dict import DictAdapter
from config.simulation_config import SimulationConfig


def from_toml(path: str) -> SimulationConfig:
    """Load a TOML file and return a validated :class:`SimulationConfig`.

    This is a convenience wrapper that uses :class:`TomlAdapter`
    internally.  The returned object is a *config* — pass it to
    :func:`setup.simulation_setup.build_setup` to obtain the
    JAX-friendly :class:`SimulationSetup`.

    Args:
        path: Filesystem path to a ``.toml`` file.

    Returns:
        A :class:`SimulationConfig`.
    """
    # Lazy import to avoid pulling in tomllib when unused
    from config.adapter_toml import TomlAdapter

    return TomlAdapter().load(path)


def from_dict(d: dict[str, Any]) -> SimulationConfig:
    """Build a :class:`SimulationConfig` from a plain dict.

    Args:
        d: Configuration mapping.

    Returns:
        A validated :class:`SimulationConfig`.
    """
    return DictAdapter().load(d)


__all__ = [
    "DictAdapter",
    "SimulationConfig",
    "from_dict",
    "from_toml",
]
