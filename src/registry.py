"""Central operator registry for TUD-LBM.

Every operator (class or pure function) self-registers at import time
via the :func:`register_operator` decorator.  Consumers look up entries
via :func:`get_operators`, :func:`get_operator_names`, or
:func:`get_operator_category`.

The registry stores :class:`OperatorEntry` objects keyed by
``"{kind}:{name}"``.

Usage::

    from registry import register_operator, get_operators

    @register_operator("collision_models")
    def collide_bgk(f, feq, tau, source=None):
        ...
    collide_bgk.name = "bgk"

    # or, for classes:
    @register_operator("collision_models")
    class CollisionBGK:
        name = "bgk"
        ...

    ops = get_operators("collision_models")
    # {"bgk": OperatorEntry(name="bgk", kind="collision_models", target=...)}
"""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic
from typing import TypeVar

OperatorTarget = Callable[..., object] | type

# TypeVar for generic OperatorEntry
T = TypeVar("T")

# TypeVar used by decorators to preserve the decorated object's type
_OT = TypeVar("_OT")


@dataclass(frozen=True)
class OperatorEntry(Generic[T]):
    """A single entry in the global operator registry."""

    name: str
    kind: str
    target: T
    metadata: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

OPERATOR_REGISTRY: dict[str, OperatorEntry[object]] = {}

# Secondary index: kind → {name → OperatorEntry}.
# Maintained by register_operator; avoids O(n) scans of OPERATOR_REGISTRY.
_KIND_INDEX: dict[str, dict[str, OperatorEntry[object]]] = {}


# ---------------------------------------------------------------------------
# Core registration decorator
# ---------------------------------------------------------------------------


def register_operator(
    kind: str,
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Decorator to register a class or function in the global registry.

    The decorated object must either:
    * Have a ``name`` class/function attribute, **or**
    * Receive *name* explicitly via the keyword argument.

    Args:
        kind: Operator category, e.g. ``"collision_models"``.
        name: Optional explicit name.  Falls back to ``obj.name`` or
            ``obj.__name__``.
        **meta: Arbitrary metadata stored in
            :attr:`OperatorEntry.metadata`.

    Returns:
        A decorator that registers *obj* and returns it unchanged.

    Raises:
        ValueError: If *obj* has no discoverable name, or if the
            ``kind:name`` key is already registered.
    """

    def decorator(obj: _OT) -> _OT:
        resolved_name = name or getattr(obj, "name", None) or getattr(obj, "__name__", None)
        if not resolved_name:
            raise ValueError(
                f"{obj!r} must define 'name' or have a __name__, or pass name= to @register_operator",
            )
        key = f"{kind}:{resolved_name}"
        if key in OPERATOR_REGISTRY:
            raise ValueError(f"Duplicate operator registration: {key}")
        entry: OperatorEntry[object] = OperatorEntry(
            name=resolved_name,
            kind=kind,
            target=obj,
            metadata=meta or None,
        )
        OPERATOR_REGISTRY[key] = entry
        _KIND_INDEX.setdefault(kind, {})[resolved_name] = entry
        return obj

    return decorator


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_operators(kind: str) -> dict[str, OperatorEntry]:
    """Return all registered operators of the given *kind*.

    Args:
        kind: Category string, e.g. ``"collision_models"``.

    Returns:
        ``{name: OperatorEntry, ...}``
    """
    return dict(_KIND_INDEX.get(kind, {}))


def get_operator_names(kind: str) -> set[str]:
    """Return the set of registered operator names for *kind*."""
    return set(_KIND_INDEX.get(kind, {}).keys())


def get_operator_category() -> set[str]:
    """Return the set of all registered operator kinds."""
    return set(_KIND_INDEX.keys())


def unregister_operator(kind: str, name: str) -> None:
    """Remove an operator from the registry (for testing only).

    Args:
        kind: Operator category, e.g. ``"collision_models"``.
        name: Operator name within that category.
    """
    key = f"{kind}:{name}"
    OPERATOR_REGISTRY.pop(key, None)
    sub = _KIND_INDEX.get(kind)
    if sub is not None:
        sub.pop(name, None)
        if not sub:
            del _KIND_INDEX[kind]


# ---------------------------------------------------------------------------
# Convenience per-kind decorators
# ---------------------------------------------------------------------------


def collision_model(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Register a collision operator (kind ``"collision_models"``)."""
    return register_operator("collision_models", name=name, **meta)


def force_model(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Register a force operator (kind ``"force"``)."""
    return register_operator("force", name=name, **meta)


def boundary_condition(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Register a boundary-condition operator (kind ``"boundary_condition"``)."""
    return register_operator("boundary_condition", name=name, **meta)


def macroscopic_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Register a macroscopic operator (kind ``"macroscopic"``)."""
    return register_operator("macroscopic", name=name, **meta)


def initialise_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Register an initialisation operator (kind ``"initialise"``)."""
    return register_operator("initialise", name=name, **meta)


def equilibrium_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Register an equilibrium operator (kind ``"equilibrium"``)."""
    return register_operator("equilibrium", name=name, **meta)


def simulation_type_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Register a simulation type (kind ``"simulation_type"``)."""
    return register_operator("simulation_type", name=name, **meta)


def stream_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Register a streaming operator (kind ``"stream"``)."""
    return register_operator("stream", name=name, **meta)


def update_timestep_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Register an update-timestep operator (kind ``"update_timestep"``)."""
    return register_operator("update_timestep", name=name, **meta)


def wetting_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Register a wetting operator (kind ``"wetting"``)."""
    return register_operator("wetting", name=name, **meta)


def lattice_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Register a lattice model (kind ``"lattice"``)."""
    return register_operator("lattice", name=name, **meta)


def plotting_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[_OT], _OT]:
    """Register a plotting operator (kind ``"plotting"``)."""
    return register_operator("plotting", name=name, **meta)
