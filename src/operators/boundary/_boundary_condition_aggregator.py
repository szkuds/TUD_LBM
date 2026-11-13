"""Composite boundary operator — the interior obstacle, then each edge's BC, in sequence.

Registers nothing: :func:`src.operators.boundary.build_bc` binds the registered
per-edge operators and hands them to :class:`CompositeBoundary`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import jax.numpy as jnp

if TYPE_CHECKING:
    from src.config.boundary import Edge
    from src.lattice.lattice import Lattice
    from src.protocols import BoundaryCondition


@dataclass(frozen=True, eq=False)
class CompositeBoundary:
    """The boundary operator of a run: the obstacle bounce-back, then every bound BC in order.

    A plain callable, deliberately not a pytree (a NamedTuple would be): JAX
    must treat it as an opaque function, so ``jax.jit`` can weak-reference it
    and flattening :class:`~src.pipeline.setup.SimulationSetup` keeps its identity.
    ``eq=False`` keeps identity hashing, since the lattice holds unhashable arrays.

    Attributes:
        edges: ``(edge, bc)`` pairs, applied in order; each *bc* already has
            its configured parameters bound.
        lattice: :class:`~src.lattice.lattice.Lattice`.
        obstacle_mask: Boolean solid-cell mask, shape ``(nx, ny, nz, 1, 1)``;
            ``None`` means the run has no obstacle.
    """

    edges: tuple[tuple[Edge, BoundaryCondition], ...]
    lattice: Lattice
    obstacle_mask: jnp.ndarray | None = None

    def __call__(self, f_stream: jnp.ndarray, f_col: jnp.ndarray, /) -> jnp.ndarray:
        """Apply all boundary conditions in sequence.

        The obstacle's full bounce-back runs before the edge BCs, so an edge BC
        sees the bounced populations next to a solid cell.

        Args:
            f_stream: Post-streaming populations.
            f_col: Post-collision populations.

        Returns:
            Populations with boundary conditions applied.
        """
        f = f_stream
        if self.obstacle_mask is not None:
            f = jnp.where(self.obstacle_mask, f_col[..., self.lattice.opp_indices, :], f)
        for edge, bc in self.edges:
            f = bc(f, f_col, self.lattice, edge)
        return f
