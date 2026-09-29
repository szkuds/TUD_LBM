"""Initialisation operators — implementations of InitialiserOperator protocol.

Public API: build_initialise_fn()

Implementation modules (_standard.py, _multiphase_bubble.py, ...) are internal;
use the factory to access.

Example:
    from operators.initialise import build_initialise_fn

    init = build_initialise_fn("standard")
    f = init((64, 64, 1), lattice, density=1.0)  # grid_shape
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.registry import get_operators

if TYPE_CHECKING:
    from src.operators.protocols import InitialiserOperator

# Auto-discover and import private operator modules for registry registration.
auto_load_operators("src.operators.initialise")


def build_initialise_fn(scheme: str = "standard") -> InitialiserOperator:
    """Return an initialisation operator satisfying InitialiserOperator protocol.

    Args:
        scheme: Initialisation type name ("standard", "multiphase_bubble", ...).
                Defaults to "standard".

    Returns:
        A callable satisfying the InitialiserOperator protocol.
        Call form: ``operator(grid_shape, lattice, **kwargs) -> f``.

    Examples:
        >>> from operators.initialise import build_initialise_fn
        >>> init = build_initialise_fn("standard")
        >>> f = init((64, 64, 1), lattice, density=1.0)
    """
    return get_operators("initialise")[scheme].target


__all__ = ["build_initialise_fn"]
