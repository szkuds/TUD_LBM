"""Per-edge wetting orchestrator.

Applies wetting to a single edge of the padded array using the
transpose trick for left/right edges, then sequences ghost-cell
reconstruction followed by wetting modification.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators.wetting._ghost_reconstruction import _reconstruct_ghost_row
from src.operators.wetting._wetting_modification import _apply_wetting_modification
from src.operators.wetting._wetting_modification import wetting_regions

if TYPE_CHECKING:
    import jax.numpy as jnp
    from src.operators.wetting._wetting_modification import WettingRegion


def _oriented_ghost_row(
    gp: jnp.ndarray,
    edge: str,
    perp_start_periodic: bool,
    perp_end_periodic: bool,
) -> tuple[jnp.ndarray, int, bool]:
    """Orient *gp* so *edge*'s ghost cells lie along axis 1, and reconstruct them.

    Normalises bottom/top vs left/right by transposing, so the ghost cells are
    always a column of the returned array.

    Returns:
        ``(arr, ghost_idx, transposed)`` — the oriented array with its ghost
        column reconstructed, that column's index, and whether *arr* is ``gp.T``.
    """
    transposed = edge in ("left", "right")
    arr = gp.T if transposed else gp

    # Ghost column index in the padded array and interior neighbor offset.
    ghost_idx = 0 if edge in ("bottom", "left") else -1
    interior_offset = 1 if ghost_idx == 0 else -1

    arr = _reconstruct_ghost_row(
        arr,
        ghost_idx,
        interior_offset,
        perp_start_periodic,
        perp_end_periodic,
    )
    return arr, ghost_idx, transposed


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
) -> jnp.ndarray:
    """Apply wetting to a single edge of the padded array.

    Reconstructs the edge's ghost row from the interior, then applies the
    wetting modification to its interior portion (excluding padding corners).
    """
    arr, ghost_idx, transposed = _oriented_ghost_row(gp, edge, perp_start_periodic, perp_end_periodic)

    modified = _apply_wetting_modification(
        arr[1:-1, ghost_idx],
        rho_l,
        rho_v,
        phi_l,
        phi_r,
        d_rho_l,
        d_rho_r,
    )
    arr = arr.at[1:-1, ghost_idx].set(modified)

    return arr.T if transposed else arr


def wetting_edge_regions(
    gp: jnp.ndarray,
    edge: str,
    perp_start_periodic: bool,
    perp_end_periodic: bool,
    rho_l: float | jnp.ndarray,
    rho_v: float | jnp.ndarray,
) -> tuple[WettingRegion, WettingRegion]:
    """Return the ``(left, right)`` regions :func:`_apply_wetting_edge` modifies.

    Runs the same orientation and ghost-row reconstruction as the applicator,
    so a caller outside the step (the contact-angle overlay) sees exactly the
    region the wetting BC changes — including the density bounds each side is
    clipped to, which are measured per contact line. Indices run along the wall:
    x for bottom/top, y for left/right.
    """
    arr, ghost_idx, _ = _oriented_ghost_row(gp, edge, perp_start_periodic, perp_end_periodic)
    return wetting_regions(arr[1:-1, ghost_idx], rho_l, rho_v)
