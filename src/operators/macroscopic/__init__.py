"""Macroscopic operators — implementations of MacroscopicOperator protocol.

Public API: build_macroscopic_fn()

Implementation modules (_single_phase.py, _multiphase.py) are internal; use the factory to access.

Example:
    from operators.macroscopic import build_macroscopic_fn

    macro = build_macroscopic_fn("standard")
    rho, u = macro(f, lattice)
"""

from __future__ import annotations
from operators._loader import auto_load_operators
from operators.factory import build_operator
from operators.protocols import MacroscopicOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("operators.macroscopic")


def build_macroscopic_fn(scheme: str = "standard") -> MacroscopicOperator:
    """Return a macroscopic operator satisfying MacroscopicOperator protocol.

    Args:
        scheme: Macroscopic model name ("standard" or others).
                Defaults to "standard" (single-phase density and velocity).

    Returns:
        A callable satisfying the MacroscopicOperator protocol.
        Can be called as: operator(f, lattice, force=None) → (rho, u)

        Type-checkers see this as a MacroscopicOperator, so:
            op: MacroscopicOperator = build_macroscopic_fn("standard")

        Type-checkers will verify any use of op matches the protocol.

    Raises:
        ValueError: If scheme is not registered.

    Examples:
        >>> from operators.macroscopic import build_macroscopic_fn
        >>> macroscopic = build_macroscopic_fn("standard")
        >>> rho, u = macroscopic(f, lattice)
    """
    return build_operator("macroscopic", scheme)


__all__ = [
    "build_macroscopic_fn",  # ← Primary API (use this!)
]
