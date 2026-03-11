"""Symmetry boundary condition operator.

Applies a mirror-symmetry rule on the specified edges of the domain.
Each incoming distribution at the wall is replaced by the corresponding
mirrored distribution from the post-collision state.
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
@dataclass(eq=False)
class SymmetryBoundaryCondition(BoundaryConditionBase):
    """Mirror-symmetry boundary condition.

    Precomputes per-edge direction index mappings so that ``__call__``
    performs only JAX array operations with no Python branching.

    For *bottom* edges the symmetry mirrors vertical directions:
        - straight up ← straight down
        - diag-top-right ← diag-bottom-right
        - diag-top-left  ← diag-bottom-left

    For *top* edges a ``jnp.roll`` is used for the diagonal components
    to account for the streaming shift.

    For *left* / *right* edges horizontal directions are mirrored.
    """

    name = "symmetry"

    # Precomputed per-edge specs – each entry is a tuple of
    # (edge_name, axis, idx, sym_pairs) where sym_pairs stores
    # (dst_dir, src_dir, roll_amount) triples.
    _edge_specs: Tuple = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        """Precompute symmetry-direction pairs for every symmetry edge."""
        if self.lattice is None or self.bc_config is None:
            return

        lattice = self.lattice
        valid_edges = self._get_valid_edges()

        specs = []
        for edge, bc_type in self.bc_config.items():
            if edge not in valid_edges:
                continue

            if edge == "bottom":
                top = lattice.construct_top_indices
                bot = lattice.construct_bottom_indices
                # (dst, src, roll, axis, idx)
                sym_pairs = (
                    (int(top[0]), int(bot[0]), 0),
                    (int(top[2]), int(bot[2]), 0),   # diag_top_right ← diag_bottom_right
                    (int(top[1]), int(bot[1]), 0),   # diag_top_left  ← diag_bottom_left
                )
                specs.append(("bottom", 1, 0, sym_pairs))

            elif edge == "top":
                top = lattice.construct_top_indices
                bot = lattice.construct_bottom_indices
                sym_pairs = (
                    (int(bot[0]), int(top[0]), 0),
                    (int(bot[1]), int(top[1]), 1),    # roll +1
                    (int(bot[2]), int(top[2]), -1),   # roll -1
                )
                specs.append(("top", 1, -1, sym_pairs))

            elif edge == "left":
                right = lattice.construct_right_indices
                left = lattice.construct_left_indices
                sym_pairs = (
                    (int(right[0]), int(left[0]), 0),
                    (int(right[1]), int(left[1]), 0),
                    (int(right[2]), int(left[2]), 0),
                )
                specs.append(("left", 0, 0, sym_pairs))

            elif edge == "right":
                left = lattice.construct_left_indices
                right = lattice.construct_right_indices
                sym_pairs = (
                    (int(left[0]), int(right[0]), 0),
                    (int(left[1]), int(right[1]), 0),
                    (int(left[2]), int(right[2]), 0),
                )
                specs.append(("right", 0, -1, sym_pairs))

        self._edge_specs = tuple(specs)

    # ------------------------------------------------------------------

    @partial(jit, static_argnums=(0,))
    def __call__(
        self, f_streamed: jnp.ndarray, f_collision: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply symmetry BC on every pre-registered edge."""
        for _edge_name, axis, idx, sym_pairs in self._edge_specs:
            for dst, src, roll in sym_pairs:
                if axis == 1:
                    src_vals = f_collision[:, idx, src, 0]
                    if roll != 0:
                        src_vals = jnp.roll(src_vals, roll, axis=0)
                    f_streamed = f_streamed.at[:, idx, dst, 0].set(src_vals)
                else:
                    src_vals = f_collision[idx, :, src, 0]
                    if roll != 0:
                        src_vals = jnp.roll(src_vals, roll, axis=0)
                    f_streamed = f_streamed.at[idx, :, dst, 0].set(src_vals)
        return f_streamed


