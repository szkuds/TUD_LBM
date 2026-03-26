"""Collision operators — implementations of CollisionOperator protocol.

Public API: build_collision_fn()

Implementation modules (_bgk.py, _mrt.py) are internal; use the factory to access.

Example:
    from operators.collision import build_collision_fn
    
    collision_fn = build_collision_fn("bgk")
    f_col = collision_fn(f, feq, tau)
"""

from __future__ import annotations

from operators.protocols import CollisionOperator
from operators.factory import build_operator

# ── Private: Import implementation modules to trigger registry registration ──
from operators.collision import _bgk as _bgk_impl  # noqa: F401
from operators.collision import _mrt as _mrt_impl  # noqa: F401


def build_collision_fn(scheme: str) -> CollisionOperator:
    """Return a collision operator satisfying CollisionOperator protocol.

    Args:
        scheme: Collision model name ("bgk" or "mrt").

    Returns:
        A callable satisfying the CollisionOperator protocol.
        Can be called as: operator(f, feq, tau, source=None) → f_col
        
        Type-checkers see this as a CollisionOperator, so:
            op: CollisionOperator = build_collision_fn("bgk")
        
        Type-checkers will verify any use of op matches the protocol.

    Raises:
        ValueError: If scheme is not registered.
        
    Examples:
        >>> from operators.collision import build_collision_fn
        >>> bgk = build_collision_fn("bgk")
        >>> f_col = bgk(f, feq, tau)
    """
    return build_operator("collision_models", scheme)


__all__ = [
    "build_collision_fn",  # ← Primary API (use this!)
]
