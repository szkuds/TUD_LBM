"""Bounce-back boundary condition operator.

Applies a half-way bounce-back rule on the specified edges of the domain.
Each incoming distribution at the wall is replaced by its opposite-direction
counterpart from the post-collision state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit

from app_setup.registry import register_operator
from .base import BoundaryConditionBase


@register_operator("boundary_condition")
@dataclass
class BounceBackBoundaryCondition(BoundaryConditionBase):
    """Half-way bounce-back boundary condition.

    Precomputes the (incoming_dir, opposite_dir) pairs and the boundary
    index for each requested edge so that ``__call__`` performs only
    JAX array operations with no Python-level branching.
    """

    name = "bounce-back"

    # Precomputed per-edge data: list of (edge_axis, idx, pairs)
    # where edge_axis is 0 (x / left-right) or 1 (y / bottom-top),
    # idx is the boundary slice index (0 or -1),
    # and pairs is a tuple of (incoming, opposite) direction indices.
    _edge_specs: Tuple = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        """Precompute direction pairs for every bounce-back edge."""
        if self.lattice is None or self.bc_config is None:
            return

        lattice = self.lattice
        opp = self.opp_indices
        valid_edges = self._get_valid_edges()

        specs = []
        for edge, bc_type in self.bc_config.items():
            if edge not in valid_edges:
                continue

            if edge == "bottom":
                incoming = lattice.construct_top_indices
                axis = 1
                idx = 0
            elif edge == "top":
                incoming = lattice.construct_bottom_indices
                axis = 1
                idx = -1
            elif edge == "left":
                incoming = lattice.construct_right_indices
                axis = 0
                idx = 0
            elif edge == "right":
                incoming = lattice.construct_left_indices
                axis = 0
                idx = -1
            else:
                continue

            pairs = tuple((int(i), int(opp[i])) for i in incoming)
            specs.append((axis, idx, pairs))

        self._edge_specs = tuple(specs)

    # ------------------------------------------------------------------

    @partial(jit, static_argnums=(0,))
    def __call__(
        self, f_streamed: jnp.ndarray, f_collision: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply bounce-back on every pre-registered edge."""
        for axis, idx, pairs in self._edge_specs:
            for i, opp_i in pairs:
                if axis == 1:
                    f_streamed = f_streamed.at[:, idx, i, 0].set(
                        f_collision[:, idx, opp_i, 0]
                    )
                else:
                    f_streamed = f_streamed.at[idx, :, i, 0].set(
                        f_collision[idx, :, opp_i, 0]
                    )
        return f_streamed


