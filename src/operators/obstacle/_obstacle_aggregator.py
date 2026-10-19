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
    """Full-way bounce-back at the masked solid cells.

    Applied between collision and streaming: a solid cell does not collide,
    it sends back what streamed into it, reversed. That is a permutation of
    the cell's own populations, so the obstacle conserves mass exactly.

    The previous form, applied after streaming, wrote the solid cell's own
    reversed *post-collision* populations over the ones that had just
    streamed in from the fluid. The obstacle then destroyed mass and
    momentum: behind a cylinder in a channel ~20% of the flux vanished and
    the wake could not shed.

    A plain callable, deliberately not a pytree, for the reasons given on
    :class:`~src.operators.boundary._boundary_condition_aggregator.CompositeBoundary`.

    Attributes:
        mask: Boolean solid-cell mask, shape ``(nx, ny, nz, 1, 1)``.
        lattice: :class:`~src.lattice.lattice.Lattice`.
    """

    mask: jnp.ndarray
    lattice: Lattice

    def __call__(self, f_col: jnp.ndarray, f_pre: jnp.ndarray) -> jnp.ndarray:
        """Replace the collision at masked solid cells with a reversal.

        Args:
            f_col: Post-collision populations.
            f_pre: Pre-collision populations (the ones streamed in last step).

        Returns:
            *f_col* with the reversed pre-collision populations on solid cells.
        """
        return jnp.where(self.mask, f_pre[..., self.lattice.opp_indices, :], f_col)
