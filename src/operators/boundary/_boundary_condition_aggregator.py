"""Composite boundary operator — applies each edge's BC in sequence.

Registers nothing: :func:`src.operators.boundary.build_bc` binds the registered
per-edge operators and hands them to :class:`CompositeBoundary`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax.numpy as jnp
    from src.lattice.lattice import Lattice
    from src.operators.protocols import BoundaryCondition


@dataclass(frozen=True, eq=False)
class CompositeBoundary:
    """The boundary operator of a run: every edge's bound BC, in application order.

    A plain callable, deliberately not a pytree (a NamedTuple would be): JAX
    must treat it as an opaque function, so ``jax.jit`` can weak-reference it
    and flattening :class:`~src.pipeline.setup.SimulationSetup` keeps its identity.
    ``eq=False`` keeps identity hashing, since the lattice holds unhashable arrays.

    Attributes:
        edges: ``(edge, bc)`` pairs, applied in order; each *bc* already has
            its configured parameters bound.
        lattice: :class:`~src.lattice.lattice.Lattice`.
    """

    edges: tuple[tuple[str, BoundaryCondition], ...]
    lattice: Lattice

    def __call__(self, f_stream: jnp.ndarray, f_col: jnp.ndarray, /) -> jnp.ndarray:
        """Apply all boundary conditions in sequence.

        Args:
            f_stream: Post-streaming populations.
            f_col: Post-collision populations.

        Returns:
            Populations with boundary conditions applied.
        """
        f = f_stream
        for edge, bc in self.edges:
            f = bc(f, f_col, self.lattice, edge)
        return f
