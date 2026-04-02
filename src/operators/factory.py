"""Unified operator factory — single source of truth for operator lookup.

This module provides a generic factory function that resolves operator names
to their implementations using the central registry. All operator-specific
factories (in collision/, streaming/, etc) delegate to this generic factory.

Example:
    def build_collision_fn(scheme: str) -> CollisionOperator:
        return build_operator("collision_models", scheme)
"""

from __future__ import annotations
from registry import OperatorTarget
from registry import get_operators


def build_operator(kind: str, scheme: str) -> OperatorTarget:
    """Resolve operator name to implementation.

    This is the single source of truth for operator factory logic.
    All operator-specific factories delegate to this function.

    Args:
        kind: Operator kind ("collision_models", "stream", "equilibrium", etc)
        scheme: Operator name within that kind ("bgk", "standard", "wb", etc)

    Returns:
        OperatorTarget: The operator function/class satisfying the protocol

    Raises:
        ValueError: If kind is not registered or scheme is unknown

    Example:
        >>> op = build_operator("collision_models", "bgk")
        >>> result = op(f, feq, tau)
    """
    ops = get_operators(kind)

    if not ops:
        raise ValueError(f"No operators registered for kind '{kind}'")

    try:
        return ops[scheme].target
    except KeyError as exc:
        # Include available schemes in error message for better UX
        valid_schemes = ", ".join(sorted(ops.keys()))
        raise ValueError(f"Unknown {kind} scheme '{scheme}'. Valid schemes: {valid_schemes}") from exc
