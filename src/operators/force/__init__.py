"""Force operators — implementations of ForcingOperator protocol.

Public API: build_force_fn()

Implementation modules are internal; use the factory to access.

Example:
    from operators.force import build_force_fn
    
    force_fn = build_force_fn("gravity_multiphase")
    template = force_fn(grid_shape=(64, 64), force_g=0.001)
"""

from __future__ import annotations

from operators.protocols import ForcingOperator
from operators.factory import build_operator
from operators._loader import auto_load_operators

# Auto-discover and import private operator modules for registry registration
auto_load_operators('operators.force')


def build_force_fn(scheme: str) -> ForcingOperator:
    """Return a force operator satisfying ForcingOperator protocol.

    Args:
        scheme: Force model name ("gravity_multiphase", "electric", etc).

    Returns:
        A callable satisfying the ForcingOperator protocol.
        
        Type-checkers see this as a ForcingOperator.

    Raises:
        ValueError: If scheme is not registered.
        
    Examples:
        >>> from operators.force import build_force_fn
        >>> gravity = build_force_fn("gravity_multiphase")
        >>> template = gravity(grid_shape=(64, 64), force_g=0.001)
    """
    # Lazy import to avoid circular dependencies
    from operators.force import electric as _elec_impl  # noqa: F401
    from operators.force import gravity as _grav_impl  # noqa: F401
    
    return build_operator("force_model", scheme)


__all__ = ["build_force_fn"]
