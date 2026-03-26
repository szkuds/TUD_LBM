"""Differential operators — implementations of DifferentialOperator protocol.

Public API: build_differential_fn()

Implementation modules are internal; use the factory to access.

Example:
    from operators.differential import build_differential_fn
    
    grad_op = build_differential_fn("gradient")
    grad_field = grad_op(scalar_field, lattice, pad_mode=[...])
"""

from __future__ import annotations

from operators.protocols import DifferentialOperator
from operators.factory import build_operator


def build_differential_fn(scheme: str) -> DifferentialOperator:
    """Return a differential operator satisfying DifferentialOperator protocol.

    Args:
        scheme: Differential operator name ("gradient", "laplacian", etc).

    Returns:
        A callable satisfying the DifferentialOperator protocol.
        
        Type-checkers see this as a DifferentialOperator.

    Raises:
        ValueError: If scheme is not registered.
        
    Examples:
        >>> from operators.differential import build_differential_fn
        >>> grad = build_differential_fn("gradient")
        >>> grad_field = grad(scalar_field, lattice, pad_mode=[...])
    """
    # Lazy import to avoid circular dependencies
    from operators.differential import _gradient as _grad_impl  # noqa: F401
    from operators.differential import _laplacian as _lap_impl  # noqa: F401
    
    return build_operator("differential", scheme)


__all__ = ["build_differential_fn"]
