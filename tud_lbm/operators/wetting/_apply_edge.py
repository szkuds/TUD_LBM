"""Per-edge wetting orchestrator.

Applies wetting to a single edge of the padded array using the
transpose trick for left/right edges, then sequences ghost-cell
reconstruction followed by wetting modification.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from tud_lbm.operators.wetting._ghost_reconstruction import _reconstruct_ghost_row
from tud_lbm.operators.wetting._wetting_modification import _apply_wetting_modification

if TYPE_CHECKING:
    import jax.numpy as jnp


def _apply_wetting_edge(
    gp: jnp.ndarray,
    edge: str,
    perp_start_periodic: bool,
    perp_end_periodic: bool,
    rho_l: jnp.ndarray,
    rho_v: jnp.ndarray,
    phi_l: jnp.ndarray,
    phi_r: jnp.ndarray,
    d_rho_l: jnp.ndarray,
    d_rho_r: jnp.ndarray,
    width: int,
) -> jnp.ndarray:
    """Apply wetting to a single edge of the padded array.

    Normalises bottom/top vs left/right by transposing so the ghost
    cells are always along axis 1 (columns), then delegates to the
    canonical row-based functions.
    """
    transposed = edge in ("left", "right")
    arr = gp.T if transposed else gp

    # Ghost column index in the padded array and interior neighbor offset.
    ghost_idx = 0 if edge in ("bottom", "left") else -1
    interior_offset = 1 if ghost_idx == 0 else -1

    arr = _reconstruct_and_modify(
        arr,
        ghost_idx,
        interior_offset,
        perp_start_periodic,
        perp_end_periodic,
        rho_l,
        rho_v,
        phi_l,
        phi_r,
        d_rho_l,
        d_rho_r,
        width,
    )

    return arr.T if transposed else arr


def _reconstruct_and_modify(
    arr: jnp.ndarray,
    ghost_idx: int,
    interior_offset: int,
    perp_start_periodic: bool,
    perp_end_periodic: bool,
    rho_l: jnp.ndarray,
    rho_v: jnp.ndarray,
    phi_l: jnp.ndarray,
    phi_r: jnp.ndarray,
    d_rho_l: jnp.ndarray,
    d_rho_r: jnp.ndarray,
    width: int,
) -> jnp.ndarray:
    """Reconstruct ghost row from interior, then apply wetting modification."""
    arr = _reconstruct_ghost_row(
        arr,
        ghost_idx,
        interior_offset,
        perp_start_periodic,
        perp_end_periodic,
    )

    # Extract the interior portion of the ghost row (exclude padding corners)
    edge_slice = arr[1:-1, ghost_idx]

    modified = _apply_wetting_modification(
        edge_slice,
        rho_l,
        rho_v,
        phi_l,
        phi_r,
        d_rho_l,
        d_rho_r,
        width,
    )

    return arr.at[1:-1, ghost_idx].set(modified)
