"""Differential operators — implementations of DifferentialOperator protocol.

Public API: build_differential_fn()

Implementation modules are internal; use the factory to access.

Example:
    from operators.differential import build_differential_fn

    grad = build_differential_fn("gradient")
    result = grad(grid, w, c, pad_modes)

Note: DifferentialConfig and DifferentialOperators are kept for backward
compatibility with tests. New code should use build_wetting_differential_operators
from the wetting module.
"""

from __future__ import annotations
from operators._loader import auto_load_operators
from operators.factory import build_operator
from operators.protocols import DifferentialOperator

# Auto-discover and import private operator modules for registry registration.
auto_load_operators("operators.differential")


def build_differential_fn(scheme: str) -> DifferentialOperator:
    """Return a differential operator satisfying DifferentialOperator protocol.

    Args:
        scheme: Differential operator name.

    Returns:
        A callable satisfying the DifferentialOperator protocol.

    Raises:
        ValueError: If scheme is not registered.
    """
    return build_operator("differential", scheme)


__all__ = [
    "build_differential_fn",
]
