"""Equilibrium operators — implementations of EquilibriumOperator protocol.

Public API: build_equilibrium_fn()

Implementation modules (_equilibrium.py) are internal; use the factory to access.

Example:
    from operators.equilibrium import build_equilibrium_fn
    
    eq = build_equilibrium_fn("wb")
    feq = eq(rho, u, lattice)
"""

from __future__ import annotations

from operators.protocols import EquilibriumOperator
from operators.factory import build_operator

# ── Private: Import implementation module to trigger registry registration ──
from operators.equilibrium import _equilibrium as _eq_impl  # noqa: F401


def build_equilibrium_fn(scheme: str = "wb") -> EquilibriumOperator:
    """Return an equilibrium operator satisfying EquilibriumOperator protocol.

    Args:
        scheme: Equilibrium model name ("wb" or others).
                Defaults to "wb" (Chai et al. D2Q9 model).

    Returns:
        A callable satisfying the EquilibriumOperator protocol.
        Can be called as: operator(rho, u, lattice) → feq
        
        Type-checkers see this as an EquilibriumOperator, so:
            op: EquilibriumOperator = build_equilibrium_fn("wb")
        
        Type-checkers will verify any use of op matches the protocol.

    Raises:
        ValueError: If scheme is not registered.
        
    Examples:
        >>> from operators.equilibrium import build_equilibrium_fn
        >>> equilibrium = build_equilibrium_fn("wb")
        >>> feq = equilibrium(rho, u, lattice)
    """
    return build_operator("equilibrium", scheme)


__all__ = [
    "build_equilibrium_fn",  # ← Primary API (use this!)
]
