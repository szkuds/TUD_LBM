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
from typing import Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

OperatorKind = Literal[
    "boundary_condition",
    "collision_models",
    "differential",
    "equilibrium",
    "force",
    "initialise",
    "lattice",
    "macroscopic",
    "plotting",
    "simulation_type",
    "stream",
    "update_timestep",
    "wetting",
]

OperatorTarget = Callable[..., object] | type


@dataclass(frozen=True)
class OperatorEntry:
    """A single entry in the global operator registry."""

    name: str
    kind: str
    target: OperatorTarget
    metadata: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

OPERATOR_REGISTRY: dict[str, OperatorEntry] = {}


# ---------------------------------------------------------------------------
# Core registration decorator
# ---------------------------------------------------------------------------


def register_operator(
    kind: str,
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
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

    def decorator(obj: OperatorTarget) -> OperatorTarget:
        resolved_name = name or getattr(obj, "name", None) or getattr(obj, "__name__", None)
        if not resolved_name:
            raise ValueError(
                f"{obj!r} must define 'name' or have a __name__, or pass name= to @register_operator",
            )
        key = f"{kind}:{resolved_name}"
        if key in OPERATOR_REGISTRY:
            raise ValueError(f"Duplicate operator registration: {key}")
        OPERATOR_REGISTRY[key] = OperatorEntry(
            name=resolved_name,
            kind=kind,
            target=obj,
            metadata=meta or None,
        )
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
    # TODO: This checks every key there should be a better way to get the desired information.
    prefix = f"{kind}:"
    return {entry.name: entry for key, entry in OPERATOR_REGISTRY.items() if key.startswith(prefix)}


def get_operator_names(kind: str) -> set[str]:
    """Return the set of registered operator names for *kind*."""
    return set(get_operators(kind).keys())


def get_operator_category() -> set[str]:
    """Return the set of all registered operator kinds."""
    return {entry.kind for entry in OPERATOR_REGISTRY.values()}


# ---------------------------------------------------------------------------
# Convenience per-kind decorators
# ---------------------------------------------------------------------------


def collision_model(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
    """Register a collision operator (kind ``"collision_models"``)."""
    return register_operator("collision_models", name=name, **meta)


def force_model(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
    """Register a force operator (kind ``"force"``)."""
    return register_operator("force", name=name, **meta)


def boundary_condition(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
    """Register a boundary-condition operator (kind ``"boundary_condition"``)."""
    return register_operator("boundary_condition", name=name, **meta)


def macroscopic_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
    """Register a macroscopic operator (kind ``"macroscopic"``)."""
    return register_operator("macroscopic", name=name, **meta)


def initialise_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
    """Register an initialisation operator (kind ``"initialise"``)."""
    return register_operator("initialise", name=name, **meta)


def equilibrium_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
    """Register an equilibrium operator (kind ``"equilibrium"``)."""
    return register_operator("equilibrium", name=name, **meta)


def simulation_type_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
    """Register a simulation type (kind ``"simulation_type"``)."""
    return register_operator("simulation_type", name=name, **meta)


def stream_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
    """Register a streaming operator (kind ``"stream"``)."""
    return register_operator("stream", name=name, **meta)


def update_timestep_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
    """Register an update-timestep operator (kind ``"update_timestep"``)."""
    return register_operator("update_timestep", name=name, **meta)


def wetting_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
    """Register a wetting operator (kind ``"wetting"``)."""
    return register_operator("wetting", name=name, **meta)


def lattice_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
    """Register a lattice model (kind ``"lattice"``)."""
    return register_operator("lattice", name=name, **meta)


def plotting_operator(
    *,
    name: str | None = None,
    **meta: object,
) -> Callable[[OperatorTarget], OperatorTarget]:
    """Register a plotting operator (kind ``"plotting"``)."""
    return register_operator("plotting", name=name, **meta)
