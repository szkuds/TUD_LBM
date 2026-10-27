"""Features — cross-cutting behaviour contributed through fixed slots.

Public API: build_features()

Each ``_*.py`` module registers one feature builder under the ``feature``
registry kind, ``build(config, lattice) -> Contribution | None``, returning
``None`` when the run does not use it. The slots and their merge rules are
documented in :mod:`src.features.contribution`.

Example:
    from src.features import build_features

    features = build_features(config, lattice)   # (("wetting", Contribution(...)),)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.registry import get_operators

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig
    from src.features.contribution import Contribution
    from src.lattice.lattice import Lattice

auto_load_operators("src.features")


def build_features(config: SimulationConfig, lattice: Lattice) -> tuple[tuple[str, Contribution], ...]:
    """Build every registered feature this run uses, in name order.

    Args:
        config: A validated :class:`~src.config.simulation_config.SimulationConfig`.
        lattice: The run's :class:`~src.lattice.lattice.Lattice`.

    Returns:
        ``(name, contribution)`` pairs; the name is the feature's ``State.ext`` key.
    """
    built = ((name, entry.target(config, lattice)) for name, entry in sorted(get_operators("feature").items()))
    return tuple((name, contribution) for name, contribution in built if contribution is not None)


__all__ = [
    "build_features",
]
