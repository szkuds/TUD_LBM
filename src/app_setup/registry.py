"""Global operator registry for TUD-LBM.

Every operator (collision_models, force, macroscopic, simulation_type, update_timestep, initialise,
etc.) registers itself here via the :func:`register_operator` class decorator.

Per-kind look-ups (e.g. "all collision_models simulation_operators") are derived dynamically
from the single global registry.  Adding a new operator only requires
defining the class with a ``name`` attribute and applying the decorator —
no central code changes are needed.

Usage::

    from registry import register_operator, get_operators, OPERATOR_REGISTRY

    @register_operator("collision_models")
    class CollisionBGK(CollisionBase):
        name = "bgk"
        ...

    # Look up all collision_models simulation_operators
    collision_ops = get_operators("collision_models")

    # Get a specific operator class
    cls = collision_ops["bgk"].cls
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar, Type, Dict

T = TypeVar("T")


@dataclass(frozen=True)
class OperatorEntry:
    """A single entry in the global operator registry."""

    name: str
    kind: str  # e.g. "collision_models", "force", "macroscopic", "simulation_type", …
    cls: Type


# Global registry of *all* simulation_operators (all kinds)
OPERATOR_REGISTRY: Dict[str, OperatorEntry] = {}


def register_operator(kind: str) -> Callable[[Type[T]], Type[T]]:
    """Class decorator to register an operator in the global registry.

    The decorated class **must** define a ``name`` class attribute.

    Args:
        kind: Operator category, e.g. ``"collision_models"``, ``"force"``,
              ``"macroscopic"``, ``"simulation_type"``, ``"update_timestep"``, …

    Returns:
        A class decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If the class has no ``name`` attribute or if the
            ``kind:name`` key is already registered.

    Example::

        @register_operator("collision_models")
        class CollisionBGK(CollisionBase):
            name = "bgk"
            ...
    """

    def decorator(cls: Type[T]) -> Type[T]:
        name = getattr(cls, "name", None)
        if not name:
            raise ValueError(f"{cls.__name__} must define a class attribute 'name'")
        key = f"{kind}:{name}"
        if key in OPERATOR_REGISTRY:
            raise ValueError(f"Duplicate operator registration: {key}")
        OPERATOR_REGISTRY[key] = OperatorEntry(name=name, kind=kind, cls=cls)
        return cls

    return decorator


def get_operators(kind: str) -> Dict[str, OperatorEntry]:
    """Return all simulation_operators of a given kind, indexed by operator name.

    Args:
        kind: The operator category to filter by.

    Returns:
        Dict mapping operator ``name`` → :class:`OperatorEntry`.

    Example::

        collision_ops = get_operators("collision_models")
        names = sorted(collision_ops.keys())        # ["bgk", "mrt"]
        cls   = collision_ops["bgk"].cls             # CollisionBGK
    """
    prefix = f"{kind}:"
    return {
        entry.name: entry
        for key, entry in OPERATOR_REGISTRY.items()
        if key.startswith(prefix)
    }

