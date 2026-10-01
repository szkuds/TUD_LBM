"""Equilibrium operators — implementations of EquilibriumOperator protocol.

Public API: build_equilibrium_fn(), build_improved_equilibrium_fn()

Implementation modules (_equilibrium.py) are internal; use the factory to access.

Example:
    from operators.equilibrium import build_equilibrium_fn

    eq = build_equilibrium_fn("wb")
    feq = eq(rho, u, lattice)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.operators.equilibrium._equilibrium_improved_well_balanced import build_improved_equilibrium
from src.registry import get_operators

if TYPE_CHECKING:
    import jax.numpy as jnp
    from src.config.viscosity_params import ViscosityParams
    from src.operators.protocols import DifferentialOperator
    from src.operators.protocols import EquilibriumOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.operators.equilibrium")


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

    Examples:
        >>> from operators.equilibrium import build_equilibrium_fn
        >>> equilibrium = build_equilibrium_fn("wb")
        >>> feq = equilibrium(rho, u, lattice)
    """
    return get_operators("equilibrium")[scheme].target


def build_improved_equilibrium_fn(
    *,
    viscosity: ViscosityParams | None,
    reference_pressure: jnp.ndarray | None,
    gradient: DifferentialOperator,
) -> EquilibriumOperator:
    """Return the improved well-balanced equilibrium of Zhang, Guo & Wang (2022).

    The ``wb`` equilibrium plus the viscous-stress term ``cs2 rho A S`` (when
    *viscosity* is given) and the reference pressure ``p_g I`` (when
    *reference_pressure* is given), bound in at setup time.

    Example:
        >>> equilibrium = build_improved_equilibrium_fn(
        ...     viscosity=config.viscosity_params, reference_pressure=None, gradient=gradient_standard
        ... )
        >>> feq = equilibrium(rho, u, lattice)
    """
    return build_improved_equilibrium(viscosity=viscosity, reference_pressure=reference_pressure, gradient=gradient)


__all__ = [
    "build_equilibrium_fn",  # ← Primary API (use this!)
    "build_improved_equilibrium_fn",
]
