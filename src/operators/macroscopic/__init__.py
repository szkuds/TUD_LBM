"""Macroscopic operators — implementations of MacroscopicOperator protocol.

Public API: build_macroscopic_fn()

The multiphase parameters these operators take are built by the configuration,
``SimulationConfig.multiphase_params`` — see :mod:`src.config.multiphase_params`.

Implementation modules (_single_phase.py, _multiphase.py) are internal; use the factory to access.

Example:
    from operators.macroscopic import build_macroscopic_fn

    macro = build_macroscopic_fn("standard")
    rho, u, _, pressure = macro(f, lattice)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.registry import get_operators

if TYPE_CHECKING:
    from src.operators.protocols import MacroscopicOperator


# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.operators.macroscopic")


def build_macroscopic_fn(scheme: str = "standard") -> MacroscopicOperator:
    """Return a macroscopic operator satisfying MacroscopicOperator protocol.

    Args:
        scheme: Macroscopic model name ("standard" or others).
                Defaults to "standard" (single-phase density and velocity).

    Returns:
        A callable satisfying the MacroscopicOperator protocol.
        Can be called as: operator(f, lattice, force=None) → (rho, u, force, pressure)

        Type-checkers see this as a MacroscopicOperator, so:
            op: MacroscopicOperator = build_macroscopic_fn("standard")

        Type-checkers will verify any use of op matches the protocol.

    Examples:
        >>> from operators.macroscopic import build_macroscopic_fn
        >>> macroscopic = build_macroscopic_fn("standard")
        >>> rho, u, _, pressure = macroscopic(f, lattice)
    """
    return get_operators("macroscopic")[scheme].target


__all__ = [
    "build_macroscopic_fn",
]
