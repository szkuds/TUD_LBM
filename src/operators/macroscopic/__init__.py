"""Macroscopic operators — implementations of MacroscopicOperator protocol.

Public API: build_macroscopic_fn(), build_multiphase_macroscopic_fn()

Both return the same :class:`~src.protocols.MacroscopicOperator`
signature; the multiphase one has the run's
:class:`~src.config.multiphase.MultiphaseParams` bound.

Example:
    from src.operators.macroscopic import build_macroscopic_fn

    macro = build_macroscopic_fn()
    rho, u, _, pressure = macro(f, lattice, None, gradient_standard=grad, laplacian_density=lap)
"""

from __future__ import annotations
import functools
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.registry import get_operators

if TYPE_CHECKING:
    from src.config.multiphase import MultiphaseParams
    from src.protocols import MacroscopicOperator


# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.operators.macroscopic")


def build_macroscopic_fn() -> MacroscopicOperator:
    """The single-phase macroscopic operator: moments and the ideal-gas pressure."""
    return get_operators("macroscopic")["standard"].target


def build_multiphase_macroscopic_fn(mp: MultiphaseParams) -> MacroscopicOperator:
    """The multiphase macroscopic operator with *mp* bound.

    Args:
        mp: The run's multiphase parameters (``src.config.multiphase.multiphase_params``).
    """
    return functools.partial(get_operators("macroscopic")["multiphase"].target, mp=mp)


__all__ = [
    "build_macroscopic_fn",
    "build_multiphase_macroscopic_fn",
]
