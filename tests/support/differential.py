"""The plain differential operators as their consumers receive them.

The macroscopic operator, the source term and the forces are handed the
gradient / Laplacian with ``State.ext`` already bound, so a test of one of them
needs ``op(grid)``, not the ``op(grid, ext)`` that ``build_differential`` builds.
"""

from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
import jax
from src.operators.differential import build_differential

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice
    from src.protocols import DifferentialOperator

PERIODIC = ("wrap", "wrap", "wrap", "wrap")


def bound_operators(
    lattice: Lattice, pad_modes: tuple[str, ...] = PERIODIC
) -> tuple[DifferentialOperator, DifferentialOperator]:
    """``(gradient, laplacian)`` without a ghost fill, jitted, with an empty ``ext`` bound."""
    gradient, laplacian = build_differential(lattice, pad_modes)
    return jax.jit(partial(gradient, ext={})), jax.jit(partial(laplacian, ext={}))
