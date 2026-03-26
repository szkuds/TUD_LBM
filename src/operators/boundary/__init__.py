"""Boundary condition operators — implementations of BoundaryOperator protocol.

Public API: build_boundary_fn()

Implementation modules (_bounce_back.py, _periodic.py, _symmetry.py) are internal.

Example:
    from operators.boundary import build_boundary_fn
    
    bc_op = build_boundary_fn("bounce-back")
    f_bc = bc_op(f_streamed, f_collision, lattice, edge="top")
"""

from __future__ import annotations

from operators.protocols import BoundaryOperator
from operators.factory import build_operator
from operators._loader import auto_load_operators

# Auto-discover and import private operator modules for registry registration
auto_load_operators('operators.boundary')


def build_boundary_fn(scheme: str = "bounce-back") -> BoundaryOperator:
    """Return a boundary condition operator satisfying BoundaryOperator protocol.

    Args:
        scheme: Boundary condition name ("bounce-back", "periodic", "symmetry", etc).
                Defaults to "bounce-back".

    Returns:
        A callable satisfying the BoundaryOperator protocol.
        Can be called as: operator(f_streamed, f_collision, lattice, edge) → f_bc
        
        Type-checkers see this as a BoundaryOperator.

    Raises:
        ValueError: If scheme is not registered.
        
    Examples:
        >>> from operators.boundary import build_boundary_fn
        >>> bc = build_boundary_fn("bounce-back")
        >>> f_bc = bc(f_streamed, f_collision, lattice, edge="top")
    """
    # Lazy import to avoid circular dependencies
    from operators.boundary import _bounce_back as _bb_impl  # noqa: F401
    from operators.boundary import _periodic as _per_impl  # noqa: F401
    from operators.boundary import _symmetry as _sym_impl  # noqa: F401
    
    return build_operator("boundary_condition", scheme)


__all__ = ["build_boundary_fn"]
