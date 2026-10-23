"""Obstacle bounce-back operator — reverses populations at solid cells.

Registers nothing: :func:`src.operators.obstacle.build_obstacle_fn` hands the
configuration's mask to :class:`ObstacleBounceBack`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import jax.numpy as jnp

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice


@dataclass(frozen=True, eq=False)
class ObstacleBounceBack:
    """Full bounce-back at the masked solid cells.

    A plain callable, deliberately not a pytree, for the reasons given on
    :class:`~src.operators.boundary._boundary_condition_aggregator.CompositeBoundary`.

    Attributes:
        mask: Boolean solid-cell mask, shape ``(nx, ny, nz, 1, 1)``.
        lattice: :class:`~src.lattice.lattice.Lattice`.
    """

    mask: jnp.ndarray
    lattice: Lattice

    def __call__(self, f_stream: jnp.ndarray, f_col: jnp.ndarray) -> jnp.ndarray:
        """Reverse populations at masked solid cells.

        Args:
            f_stream: Post-streaming populations.
            f_col: Post-collision populations.

        Returns:
            *f_stream* with the reversed post-collision populations on solid cells.
        """
        return jnp.where(self.mask, f_col[..., self.lattice.opp_indices, :], f_stream)
