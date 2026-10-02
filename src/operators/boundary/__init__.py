"""Boundary condition operators — implementations of BoundaryOperator protocol.

Public API: build_bc()

Implementation modules (_bounce_back.py, _outlet.py, _periodic.py, _symmetry.py,
_velocity_inlet.py) are internal; use the factory to access.

Example:
    from src.operators.boundary import build_bc

    bc_fn = build_bc(config.boundary_edges, lattice)
    f = bc_fn(f_stream, f_col)
"""

from __future__ import annotations
import functools
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.operators.boundary._boundary_condition_aggregator import CompositeBoundary
from src.registry import get_operators

if TYPE_CHECKING:
    from src.config.boundary_edges import BoundaryEdge
    from src.lattice.lattice import Lattice
    from src.operators.protocols import BoundaryOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.operators.boundary")


def build_bc(edges: tuple[BoundaryEdge, ...], lattice: Lattice) -> BoundaryOperator:
    """Return the run's boundary operator, satisfying BoundaryOperator protocol.

    Args:
        edges: Each edge's BC and parameters, in application order —
            :attr:`SimulationConfig.boundary_edges
            <src.config.simulation_config.SimulationConfig.boundary_edges>`.
        lattice: :class:`~src.lattice.lattice.Lattice`.

    Returns:
        A callable ``bc_fn(f_stream, f_col) → f``.
    """
    bc_ops = get_operators("boundary_condition")
    return CompositeBoundary(
        edges=tuple((e.edge, functools.partial(bc_ops[e.name].target, **e.params)) for e in edges),
        lattice=lattice,
    )


__all__ = [
    "build_bc",
]
