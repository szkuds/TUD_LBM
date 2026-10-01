"""Source-term operators — implementations of SourceTermOperator protocol.

Public API: build_source_fn()

Implementation modules (_source_well_balanced.py) are internal; use the factory to access.

Example:
    from operators.source_term import build_source_fn

    source_fn = build_source_fn("wb")
    src = source_fn(rho, u, force, lattice, gradient=gradient_density)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.registry import get_operators

if TYPE_CHECKING:
    from src.operators.protocols import SourceTermOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.operators.source_term")


def build_source_fn(scheme: str = "wb") -> SourceTermOperator:
    """Return a source-term operator satisfying SourceTermOperator protocol.

    Args:
        scheme: Source-term model name. Defaults to "wb" (well-balanced forcing).

    Returns:
        A callable satisfying the SourceTermOperator protocol.
        Can be called as: operator(rho, u, force, lattice, *, gradient) → src

    Examples:
        >>> from operators.source_term import build_source_fn
        >>> source_fn = build_source_fn("wb")
        >>> src = source_fn(rho, u, force, lattice, gradient=gradient_density)
    """
    return get_operators("source_term")[scheme].target


__all__ = [
    "build_source_fn",
]
