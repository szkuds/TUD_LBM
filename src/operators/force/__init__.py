"""Force operators — implementations of the force operator protocol.

Public API: build_force_fn()

Implementation modules are internal; use the factory to access.

Example:
    from operators.force import build_force_fn

    gravity = build_force_fn("gravity_force")
    template = gravity.build({"force_g": 0.001}, (64, 64))
"""

from __future__ import annotations

from typing import Callable

from operators.factory import build_operator
from operators._loader import auto_load_operators
from operators.protocols import ForceOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators('operators.force')


def build_force_fn(scheme: str) -> Callable[..., object] | type:
    """Return a registered force module.

    Args:
        scheme: Force model name ("gravity_force", "electric_force", etc).

    Returns:
        A registry-backed force module exposing ``build`` and ``compute``.

    Raises:
        ValueError: If scheme is not registered.

    Examples:
        >>> from operators.force import build_force_fn
        >>> gravity = build_force_fn("gravity_force")
        >>> template = gravity.build({"force_g": 0.001}, (64, 64))
    """
    # Lazy imports trigger module registration via decorators.
    from operators.force import _electric as _elec_impl  # noqa: F401
    from operators.force import _gravity as _grav_impl  # noqa: F401

    return build_operator("force", scheme)


__all__ = ["build_force_fn"]
