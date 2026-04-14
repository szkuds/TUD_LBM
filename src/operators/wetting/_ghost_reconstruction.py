"""Ghost-cell reconstruction using D2Q9 lattice stencil weights.

Provides :func:`_reconstruct_ghost_row`, which fills a ghost row of a
padded 2-D density array from its interior neighbour using a weighted
average of the cardinal and two diagonal neighbours (D2Q9 weights).
"""

from __future__ import annotations
import jax.numpy as jnp

# Stencil weights for ghost-cell reconstruction (Lattice, D2Q9)
_W_CARDINAL = 1.0 / 3.0
_W_DIAGONAL = 1.0 / 12.0
_W_TOTAL = _W_CARDINAL + 2.0 * _W_DIAGONAL


def _reconstruct_ghost_row(
    arr: jnp.ndarray,
    ghost_idx: int,
    interior_offset: int,
    wrap_start: bool,
    wrap_end: bool,
) -> jnp.ndarray:
    """Reconstruct ghost-cell values using D2Q9 stencil weights.

    The ghost row value at position ``i`` is a weighted average of the
    three nearest interior-row neighbors: the cardinal neighbor directly
    inward, and the two diagonal neighbors (i-1, i+1).

    Corner handling depends on whether the perpendicular BC is periodic.

    Args:
        arr: Padded array, shape ``(n_along_wall + 2, n_normal + 2)``.
        ghost_idx: Column index of the ghost row (0 or -1).
        interior_offset: +1 if ghost is at 0, -1 if ghost is at -1.
        wrap_start: True if the start corner (index 0) wraps periodically.
        wrap_end: True if the end corner (index -1) wraps periodically.

    Returns:
        Updated array with ghost row reconstructed.
    """
    int_col = ghost_idx + interior_offset

    # Interior points (indices 1:-1 along the wall, excluding corners)
    cardinal = arr[1:-1, int_col]
    diag_minus = arr[:-2, int_col]
    diag_plus = arr[2:, int_col]
    arr = arr.at[1:-1, ghost_idx].set(
        (_W_CARDINAL * cardinal + _W_DIAGONAL * (diag_minus + diag_plus)) / _W_TOTAL
    )

    # Start corner (index 0)
    start_wrap = arr[-1, int_col] if wrap_start else arr[1, int_col]
    arr = arr.at[0, ghost_idx].set(
        (_W_CARDINAL * arr[0, int_col] + _W_DIAGONAL * (start_wrap + arr[1, int_col])) / _W_TOTAL
    )

    # End corner (index -1)
    end_wrap = arr[0, int_col] if wrap_end else arr[-2, int_col]
    return arr.at[-1, ghost_idx].set(
        (_W_CARDINAL * arr[-1, int_col] + _W_DIAGONAL * (arr[-2, int_col] + end_wrap)) / _W_TOTAL
    )


