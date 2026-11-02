"""Differential operators — one builder for the gradient and the Laplacian.

Public API: build_differential()

Both operators are ``op(grid, ext)``, built with the lattice, the pad modes and
an optional *ghost fill* already bound. The fill is the composed ``ghost_fill``
slot of the active addons (:mod:`src.addons.hooks`): it rewrites the
padded field's ghost layer from the addon substates in *ext* between padding
and the stencil, which is how the wetting wall reaches the stencils. Without a
fill the stencil is plain and ignores *ext*. The package knows no addon by
name. The closures live in the implementation modules (_gradient.py,
_laplacian.py); there is no ``differential`` registry kind, because nothing is
selected between by configuration.

Example:
    from src.operators.differential import build_differential

    gradient, laplacian = build_differential(lattice, config.pad_modes)
    grad_rho = gradient(rho, state.ext)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators.differential._gradient import build_gradient
from src.operators.differential._laplacian import build_laplacian

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice
    from src.protocols import ExtDifferentialOperator
    from src.protocols import ExtGhostFill


def build_differential(
    lattice: Lattice, pad_modes: tuple[str, ...], ghost_fill: ExtGhostFill | None = None
) -> tuple[ExtDifferentialOperator, ExtDifferentialOperator]:
    """Return ``(gradient, laplacian)`` over *pad_modes*, with *ghost_fill* applied after padding.

    Args:
        lattice: The simulation :class:`Lattice` (weights, velocities).
        pad_modes: ``SimulationConfig.pad_modes``, ``(top, bottom, right, left)``.
        ghost_fill: The addons' composed ghost fill, or ``None`` for the plain stencils.

    Returns:
        ``grad(grid, ext) → (nx, ny, nz, 1, 2)`` and ``lap(grid, ext) → (nx, ny, nz, 1, 1)``.
    """
    return (
        build_gradient(lattice.w, lattice.c, pad_modes, ghost_fill),
        build_laplacian(lattice.w, pad_modes, ghost_fill),
    )


__all__ = [
    "build_differential",
]
