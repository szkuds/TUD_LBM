"""Composite boundary operator — the interior obstacle, then each edge's BC, in sequence.

Registers nothing: :func:`src.operators.boundary.build_bc` binds the registered
per-edge operators and hands them to :meth:`CompositeBoundary.from_parts`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import jax.numpy as jnp

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice
    from src.protocols import BoundaryCondition

#: The chain key of the interior obstacle, which is not a domain edge.
OBSTACLE = "obstacle"


@dataclass(frozen=True, eq=False)
class ObstacleBounceBack:
    """Full bounce-back at the masked solid cells: a :class:`BoundaryCondition` on no edge.

    A plain callable, deliberately not a pytree, for the reasons given on
    :class:`CompositeBoundary`.

    Attributes:
        mask: Boolean solid-cell mask, shape ``(nx, ny, nz, 1, 1)``.
    """

    mask: jnp.ndarray

    def __call__(self, f_stream: jnp.ndarray, f_col: jnp.ndarray, lattice: Lattice, edge: str, /) -> jnp.ndarray:  # noqa: ARG002 - BoundaryCondition shape
        """Reverse the post-collision populations at the masked solid cells.

        Args:
            f_stream: Post-streaming populations.
            f_col: Post-collision populations.
            lattice: :class:`~src.lattice.lattice.Lattice`.
            edge: Unused; the obstacle is interior.

        Returns:
            *f_stream* with the reversed post-collision populations on solid cells.
        """
        return jnp.where(self.mask, f_col[..., lattice.opp_indices, :], f_stream)


@dataclass(frozen=True, eq=False)
class CompositeBoundary:
    """The boundary operator of a run: every bound BC, in application order.

    A plain callable, deliberately not a pytree (a NamedTuple would be): JAX
    must treat it as an opaque function, so ``jax.jit`` can weak-reference it
    and flattening :class:`~src.pipeline.setup.SimulationSetup` keeps its identity.
    ``eq=False`` keeps identity hashing, since the lattice holds unhashable arrays.

    Attributes:
        edges: ``(edge, bc)`` pairs, applied in order; each *bc* already has
            its configured parameters bound. An interior obstacle comes first,
            under the key :data:`OBSTACLE`.
        lattice: :class:`~src.lattice.lattice.Lattice`.
    """

    edges: tuple[tuple[str, BoundaryCondition], ...]
    lattice: Lattice

    @classmethod
    def from_parts(
        cls,
        edges: tuple[tuple[str, BoundaryCondition], ...],
        lattice: Lattice,
        obstacle_mask: jnp.ndarray | None,
    ) -> CompositeBoundary:
        """The chain for *edges*, headed by the obstacle bounce-back when *obstacle_mask* is given.

        The obstacle runs before the edge BCs, so an edge BC sees the bounced
        populations next to a solid cell, as it always has.
        """
        if obstacle_mask is None:
            return cls(edges=edges, lattice=lattice)
        return cls(edges=((OBSTACLE, ObstacleBounceBack(mask=obstacle_mask)), *edges), lattice=lattice)

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
