"""Addons — cross-cutting behaviour contributed through fixed slots.

Public API: build_addons()

Each ``_*.py`` module registers one addon builder under the ``addon``
registry kind, ``build(config) -> Hooks | None``, returning
``None`` when the run does not use it. The slots and their merge rules are
documented in :mod:`src.addons.hooks`.

Example:
    from src.addons import build_addons

    addons = build_addons(config)   # (("wetting", Hooks(...)),)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.registry import get_operators

if TYPE_CHECKING:
    from src.addons.hooks import Hooks
    from src.config.simulation_config import SimulationConfig

auto_load_operators("src.addons")


def build_addons(config: SimulationConfig) -> tuple[tuple[str, Hooks], ...]:
    """Build every registered addon this run uses, in name order.

    Args:
        config: A validated :class:`~src.config.simulation_config.SimulationConfig`.

    Returns:
        ``(name, hooks)`` pairs; the name is the addon's ``State.ext`` key.
    """
    built = ((name, entry.target(config)) for name, entry in sorted(get_operators("addon").items()))
    return tuple((name, hooks) for name, hooks in built if hooks is not None)


__all__ = [
    "build_addons",
]
