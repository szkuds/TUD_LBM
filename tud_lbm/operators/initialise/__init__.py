"""Initialisation operators — implementations of InitialiserOperator protocol.

Public API: build_initialise_fn()

Implementation modules (_standard.py, _multiphase_bubble.py, ...) are internal;
use the factory to access.

Example:
    from operators.initialise import build_initialise_fn

    init = build_initialise_fn("standard")
    f = init(64, 64, 1, lattice, density=1.0)  # (nx, ny, nz)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from tud_lbm.operators._loader import auto_load_operators
from tud_lbm.operators.factory import build_operator

if TYPE_CHECKING:
    from tud_lbm.operators.protocols import InitialiserOperator

# Auto-discover and import private operator modules for registry registration.
auto_load_operators("tud_lbm.operators.initialise")


def build_initialise_fn(scheme: str = "standard") -> InitialiserOperator:
    """Return an initialisation operator satisfying InitialiserOperator protocol.

    Args:
        scheme: Initialisation type name ("standard", "multiphase_bubble", ...).
                Defaults to "standard".

    Returns:
        A callable satisfying the InitialiserOperator protocol.
        Can be called as: operator(nx, ny, nz, lattice, **kwargs) -> f

    Raises:
        ValueError: If scheme is not registered.

    Examples:
        >>> from operators.initialise import build_initialise_fn
        >>> init = build_initialise_fn("standard")
        >>> f = init(64, 64, 1, lattice, density=1.0)
    """
    return build_operator("initialise", scheme)


__all__ = ["build_initialise_fn"]
