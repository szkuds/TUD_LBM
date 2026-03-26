"""Macroscopic operators — implementations of MacroscopicOperator protocol.

Public API: build_macroscopic_fn()

Implementation modules (_single_phase.py, _multiphase.py) are internal; use the factory to access.

Example:
    from operators.macroscopic import build_macroscopic_fn
    
    macro = build_macroscopic_fn("standard")
    rho, u = macro(f, lattice)
"""

from __future__ import annotations

from operators.protocols import MacroscopicOperator
from operators.factory import build_operator

# ── Private: Import implementation modules to trigger registry registration ──
from operators.macroscopic import _single_phase as _sp_impl  # noqa: F401
from operators.macroscopic import _multiphase as _mp_impl    # noqa: F401


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
