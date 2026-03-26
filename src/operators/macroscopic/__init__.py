"""Macroscopic operators — implementations of MacroscopicOperator protocol.

RECOMMENDED USAGE:
    from operators.macroscopic import build_macroscopic_fn
    from operators.protocols import MacroscopicOperator
    
    macro: MacroscopicOperator = build_macroscopic_fn("standard")
    rho, u = macro(f, lattice)

The factory function (build_macroscopic_fn) is the stable public API.
Implementation modules (_single_phase.py, _multiphase.py) are internal details.

For extending with your own macroscopic operator:
    1. Implement a function matching MacroscopicOperator protocol
    2. Register it with @macroscopic_model(name="your_name")
    3. Access via build_macroscopic_fn("your_name")
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
