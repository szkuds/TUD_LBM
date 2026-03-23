"""Collision operator factory — string name → pure function lookup.

Uses the global operator registry to resolve collision model names
to their corresponding pure functions.
"""

from __future__ import annotations
from collections.abc import Callable
from registry import get_operators


def build_collision_fn(scheme: str) -> Callable:
    """Return the collision pure function for *scheme*.

    Args:
        scheme: Collision model name (``"bgk"`` or ``"mrt"``).

    Returns:
        A callable ``(f, feq, tau, source=None, ...) → f_col``.

    Raises:
        ValueError: If *scheme* is not registered.
    """
    collision_ops = get_operators("collision_models")
    try:
        entry = collision_ops[scheme]
    except KeyError as exc:
        valid = ", ".join(sorted(collision_ops))
        raise ValueError(
            f"Unknown collision scheme '{scheme}'. Valid: {valid}",
        ) from exc
    return entry.target
