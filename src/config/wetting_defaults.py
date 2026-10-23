"""Wetting defaults — the four wetting scalars of a configuration.

Resolved only by :attr:`SimulationConfig.wetting_defaults
<src.config.simulation_config.SimulationConfig.wetting_defaults>`: the wetting
state initialiser and the wetting-aware differential operators both start from
these values, so they are read once, here, rather than by each consumer.
"""

from __future__ import annotations
from typing import Any

#: The four wetting scalars, each as ``(canonical_key, legacy_key, default)``.
#:
#: One table rather than a default dict plus a per-parameter default: the two
#: drifted apart the moment they were written twice, and nothing checks that a
#: neutral config built in one module agrees with a reader defaulting in another.
WETTING_SCALARS: tuple[tuple[str, str, float], ...] = (
    ("phi_left", "phi_l", 1.0),
    ("phi_right", "phi_r", 1.0),
    ("d_rho_left", "d_rho_l", 0.0),
    ("d_rho_right", "d_rho_r", 0.0),
)

#: Wetting configuration with no wall modification, for a run that asks for
#: hysteresis without declaring a ``[wetting]`` section.
NEUTRAL_WETTING_CONFIG: dict[str, float] = {name: default for name, _legacy, default in WETTING_SCALARS}


def wetting_scalar(cfg: dict[str, Any], name: str, legacy_name: str, *, default: float) -> float:
    """Read one wetting scalar, accepting the legacy short key.

    ``wetting_config`` is a free-form ``dict[str, Any]`` straight off the TOML,
    so a lookup is ``Any`` and a missing key is ``None`` -- neither of which
    ``float()`` accepts. A key present but null means the default.
    """
    for key in (name, legacy_name):
        value = cfg.get(key)
        if value is not None:
            return float(value)
    return default


def resolve_wetting_defaults(wetting_config: dict[str, Any]) -> dict[str, float]:
    """The four wetting scalars of *wetting_config*, keyed by canonical name."""
    return {
        name: wetting_scalar(wetting_config, name, legacy, default=default) for name, legacy, default in WETTING_SCALARS
    }
